from . import learn
from .. import settings
from ..settings import validation
import torch
import optuna
import dataclasses
import copy
import os
import pickle
import contextlib
import multiprocessing


from typing import TypeVar, Any, overload, Sequence, Generator, Literal


SAMPLE_LOG = ("lr", )


class GpuQueue:
    # From https://vordeck.de/kn/optuna-gpu-queue
    def __init__(self):
        self.queue = multiprocessing.Manager().Queue()
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} device(s).")
        self.all_idxs = list(
            range(device_count)) if device_count > 0 else ["cpu"]
        for idx in self.all_idxs:
            self.queue.put(idx)

    @contextlib.contextmanager
    def one_gpu_per_process(self) -> Generator[
            int | Literal["cpu"], None, None]:
        current_idx = self.queue.get()
        yield current_idx
        self.queue.put(current_idx)


@overload
def suggest_numerical(
        trial: optuna.Trial,
        name: str,
        selection: tuple[int, int]
        ) -> int:
    ...


@overload
def suggest_numerical(
        trial: optuna.Trial,
        name: str,
        selection: tuple[float, float]
        ) -> float:
    ...


def suggest_numerical(
        trial: optuna.Trial,
        name: str,
        selection: tuple[int, int] | tuple[float, float]
        ) -> int | float:
    if isinstance(selection[0], int):
        assert isinstance(selection[1], int)
        return trial.suggest_int(
            name, selection[0], selection[1])
    else:
        assert isinstance(selection[1], float)
        log = any([n in name for n in SAMPLE_LOG])
        return trial.suggest_float(
            name, selection[0], selection[1], log=log)


T = TypeVar("T")


def suggest_categorical(
        trial: optuna.Trial,
        name: str,
        selection: Sequence[T],
        ) -> T:
    return trial.suggest_categorical(
        name, selection  # type: ignore
    )


Value = Any | tuple[
    str, ...] | tuple[bool, ...] | tuple[int, int] | tuple[
        float, float] | dict[str, "Value"]
Sampled = Any | dict[str, "Sampled"]


def value_sample(
        trial: optuna.Trial,
        name: str,
        value: Value,
        ) -> Sampled:
    if isinstance(value, tuple):
        assert len(value) > 1
        if isinstance(value[0], (int, float)):
            assert len(value) == 2
            assert type(value[0]) is type(value[1])
            return suggest_numerical(
                trial, name, value  # type: ignore
            )
        else:
            return suggest_categorical(
                trial, name, value
            )

    elif isinstance(value, dict):
        out_dict: dict[str, Sampled] = {}
        for sub_name, sub_value in value.items():
            out_dict[sub_name] = value_sample(
                trial,
                f"{name}_{sub_name}",
                sub_value
            )
        return out_dict
    else:
        return value


def sample(
        trial: optuna.Trial,
        tagging_settings: settings.TaggingSettings,
        ranges_settings: settings.TaggingRangesSettings,
        ) -> settings.TaggingSettings:
    tagging_setts = copy.deepcopy(dataclasses.asdict(tagging_settings))
    ranges_setts = copy.deepcopy(dataclasses.asdict(ranges_settings))

    for name, value in ranges_setts.items():
        if value is not None:
            tagging_setts[name] = value_sample(
                trial, name, value
            )

    tagging_setts["mode"] = "init"
    return settings.TaggingSettings(
        **tagging_setts
    )


# TODO: CHECK whether I have to save the model to resume pruner

class Objective:
    def __init__(
            self,
            args: settings.OptSettings,
            gpu_queue: GpuQueue,
            sampler_path: str,
            pruner_path: str,
            ) -> None:
        self.gpu_queue = gpu_queue
        self.args = args

        self.sampler_path: str = sampler_path
        self.pruner_path: str = pruner_path

        self.data = learn.prepare_data_and_loaders(
            self.args.file,
            self.args.deprels,
            self.args.tagging.model_path,
            self.args.tagging.batch_size,
            get_loaders=False,
            device=torch.device("cpu"),
        )[:2]

    def save_pruner_and_sampler(
            self, trial: optuna.Trial,
            gpu_i: int | Literal["cpu"] = "cpu") -> None:
        # In multi-GPU save only if on GPU 0
        if gpu_i == 0 or gpu_i == "cpu":
            with open(self.sampler_path, "wb") as fout:
                pickle.dump(trial.study.sampler, fout)
            with open(self.pruner_path, "wb") as fout:
                pickle.dump(trial.study.pruner, fout)

    def __call__(self, trial: optuna.Trial) -> float:
        with self.gpu_queue.one_gpu_per_process() as gpu_i:
            tagging_settings = sample(
                trial,
                self.args.tagging,
                self.args.ranges)
            validation.assert_tagging_settings(
                    tagging_settings
                )
            file_settings = self.args.file
            dep_settings = self.args.deprels

            results: list[float] = []
            for step, eval_score in enumerate(
                    learn.train_command(
                        tagging_settings=tagging_settings,
                        file_settings=file_settings,
                        dep_settings=dep_settings,
                        data=(*self.data, *learn.prepare_training_loaders(
                            self.data[0], self.data[1],
                            self.args.tagging.batch_size,
                            device=torch.device(gpu_i))),
                        save_model=False,
                        device=torch.device(gpu_i),
                        final_eval=False)):
                results.append(eval_score)
                trial.report(eval_score, step)

                # Handle pruning
                if trial.should_prune():
                    self.save_pruner_and_sampler(trial, gpu_i)
                    raise optuna.TrialPruned()

            # Return best eval_score
            self.save_pruner_and_sampler(trial, gpu_i)
            return max(results)


def optimise(args: settings.OptSettings, seed: int = 1):
    pruner_path = f"./opt/{args.study_name}_pruner.pkl"
    if args.tagging.mode == "continue" and os.path.exists(
            pruner_path):
        pruner = pickle.load(open(pruner_path, "rb"))
    else:
        pruner = optuna.pruners.HyperbandPruner(
            args.pruner_min_resource,
            args.pruner_max_resource,
            args.pruner_reduction_factor,
            args.pruner_bootstrap_count,
        )

    sampler_path = f"./opt/{args.study_name}_sampler.pkl"
    if args.tagging.mode == "continue" and os.path.exists(
            sampler_path):
        sampler = pickle.load(open(sampler_path, "rb"))
    else:
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=args.sampler_n_startup_trials,
            n_ei_candidates=args.sampler_n_ei_candidates,
            seed=seed,
            multivariate=args.sampler_multivariate,
        )
    # group not needed due to no tree-structured sampling space

    storage_name = f"sqlite:///./opt/{args.study_name}.db"
    study = optuna.create_study(
        pruner=pruner,
        sampler=sampler,
        study_name=args.study_name,
        direction="maximize",
        load_if_exists=args.tagging.mode == "continue",
        storage=storage_name,
    )

    gpu_queue = GpuQueue()
    objective = Objective(
        args, gpu_queue, sampler_path, pruner_path)

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=len(gpu_queue.all_idxs))
