from . import learn, config
from .. import settings, utils
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
        try:
            yield current_idx
        finally:
            self.queue.put(current_idx)


@overload
def suggest_numerical(
        trial: optuna.Trial,
        name: str,
        selection: tuple[int, int],
        ) -> int:
    ...


@overload
def suggest_numerical(
        trial: optuna.Trial,
        name: str,
        selection: tuple[float, float],
        ) -> float:
    ...


@overload
def suggest_numerical(
        trial: optuna.Trial,
        name: str,
        selection: tuple[float, float, int],
        ) -> float:
    ...


def suggest_numerical(
        trial: optuna.Trial,
        name: str,
        selection: tuple[
            int, int] | tuple[float, float] | tuple[float, float, float],
        ) -> int | float:
    if isinstance(selection[0], int):
        assert isinstance(selection[1], int)
        return trial.suggest_int(
            name, selection[0], selection[1])
    else:
        assert isinstance(selection[1], float)
        log = any([n in name for n in SAMPLE_LOG])
        step: float | None = None
        if len(selection) == 3:
            assert isinstance(selection[2], float)
            step = selection[2]
        return trial.suggest_float(
            name, selection[0], selection[1], log=log, step=step)


T = TypeVar("T")


def suggest_categorical(
        trial: optuna.Trial,
        name: str,
        selection: Sequence[T],
        ) -> T:
    return trial.suggest_categorical(
        name, selection  # type: ignore
    )


Selection = tuple[
    str, ...] | tuple[bool, ...] | tuple[int, int] | tuple[
        float, float] | tuple[
        float, float, float] | dict[str, "Selection"]
Value = Any | tuple[
    str, ...] | tuple[bool, ...] | tuple[int, int] | tuple[
        float, float] | tuple[
        float, float, float] | dict[str, "Value"]
Sampled = Any | dict[str, "Sampled"]


def value_sample(
        trial: optuna.Trial,
        name: str,
        value: Value,
        ) -> Sampled:
    if isinstance(value, tuple):
        assert len(value) > 1
        if isinstance(value[0], bool):
            pass
        elif isinstance(value[0], (int, float)):
            assert len(value) == 2 or len(value) == 3
            assert type(value[0]) is type(value[1])
            if len(value) == 3:
                assert type(value[0]) is type(value[2])
            return suggest_numerical(
                trial, name, value,  # type: ignore
            )
        return suggest_categorical(
            trial, name, value)

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
        ranges_settings: (
            settings.TaggingRangesSettings
            | settings.TaggingEvalRangesSettings),
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
            seed=args.tagging.seed,
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
            seed = tagging_settings.seed + trial.number
            utils.set_seed(seed)

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
                            device=torch.device(gpu_i),
                            seed=seed,)),
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


def optimise(args: settings.OptSettings):
    pruner_path = f"{args.opt_path}/{args.study_name}_pruner.pkl"
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

    sampler_path = f"{args.opt_path}/{args.study_name}_sampler.pkl"
    if args.tagging.mode == "continue" and os.path.exists(
            sampler_path):
        sampler = pickle.load(open(sampler_path, "rb"))
    else:
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=args.sampler_n_startup_trials,
            n_ei_candidates=args.sampler_n_ei_candidates,
            seed=args.tagging.seed,
            multivariate=args.sampler_multivariate,
        )
    # group not needed due to no tree-structured sampling space

    storage_name = f"sqlite:///{args.opt_path}/{args.study_name}.db"
    storage = optuna.storages.RDBStorage(
        url=storage_name, engine_kwargs={"connect_args": {"timeout": 100}})
    study = optuna.create_study(
        pruner=pruner,
        sampler=sampler,
        study_name=args.study_name,
        direction="maximize",
        load_if_exists=args.tagging.mode == "continue",
        storage=storage,
    )

    gpu_queue = GpuQueue()
    objective = Objective(
        args, gpu_queue, sampler_path, pruner_path)

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=len(gpu_queue.all_idxs))


class EvalObjective:
    def __init__(
            self,
            args: settings.EvalOptSettings,
            gpu_queue: GpuQueue,
            sampler_path: str,
            ) -> None:
        self.gpu_queue = gpu_queue
        self.args = args

        self.sampler_path: str = sampler_path

        assert args.file.split is not None
        self.data = learn.prepare_data_and_loaders_eval(
            self.args.file,
            self.args.deprels,
            self.args.tagging.tag_vocab_path,
            self.args.tagging.model_path,
            self.args.tagging.batch_size,
            get_loader=False,
            split=args.file.split,
            seed=args.tagging.seed,
        )[0]

        self.tagging_model = config.initialise_model(
            self.data.sup2id,
            self.args.tagging.model_path,
            num_pos_tags=len(self.data.pos_dict),
            num_xpos_tags=len(self.data.xpos_dict),
            num_deprel_tags=len(self.data.deprel_dict),
            num_sup_deprel_tags=len(self.data.sup_deprel_dict),
            num_feats_tags={
                feat: len(dic) for feat, dic in self.data.feats_dicts.items()},
            train_deprel=self.args.tagging.train_deprel,
            train_arc=self.args.tagging.train_arc,
            train_sup=self.args.tagging.train_sup,
            train_pos=self.args.tagging.train_pos,
            train_xpos=self.args.tagging.train_xpos,
            train_feats=self.args.tagging.train_feats,
            factorised=self.args.tagging.factorised,
            extra_num_labels={
                subtype: len(dic)
                for subtype, dic
                in self.data.subtypes_dicts.items()},
            train_subtypes=self.args.tagging.train_subtypes,
            pos_label_smoothing=self.args.tagging.pos_label_smoothing,
            xpos_label_smoothing=self.args.tagging.xpos_label_smoothing,
            arc_label_smoothing=self.args.tagging.arc_label_smoothing,
            deprel_label_smoothing=self.args.tagging.deprel_label_smoothing,
            sup_label_smoothing=self.args.tagging.sup_label_smoothing,
            feats_label_smoothing=self.args.tagging.feats_label_smoothing,
            subtypes_label_smoothing=(
                self.args.tagging.subtypes_label_smoothing),
            proj_drop=self.args.tagging.proj_drop,
            arc_drop=self.args.tagging.arc_drop,
            deprel_drop=self.args.tagging.deprel_drop,
            mix_drop=self.args.tagging.mix_drop,
            deprel_hidden=self.args.tagging.deprel_hidden,
            arc_hidden=self.args.tagging.arc_hidden,
            compile=self.args.tagging.compile,)

    def save_sampler(
            self, trial: optuna.Trial,
            gpu_i: int | Literal["cpu"] = "cpu") -> None:
        # In multi-GPU save only if on GPU 0
        if gpu_i == 0 or gpu_i == "cpu":
            with open(self.sampler_path, "wb") as fout:
                pickle.dump(trial.study.sampler, fout)

    def __call__(self, trial: optuna.Trial) -> float:
        with self.gpu_queue.one_gpu_per_process() as gpu_i:
            tagging_settings = sample(
                trial,
                self.args.tagging,
                self.args.ranges)
            validation.assert_tagging_settings(
                    tagging_settings
                )
            seed = tagging_settings.seed + trial.number
            utils.set_seed(seed)

            file_settings = self.args.file
            dep_settings = self.args.deprels

            eval_score = learn.evaluate_command(
                tagging_settings=tagging_settings,
                file_settings=file_settings,
                dep_settings=dep_settings,
                data=(
                    self.data, learn.prepare_eval_loader(
                        self.data,
                        self.args.tagging.batch_size,
                        device=torch.device(gpu_i),
                        seed=seed,)),
                device=torch.device(gpu_i),
                model=self.tagging_model)
            assert eval_score is not None

            # Return best eval_score
            self.save_sampler(trial, gpu_i)
            return eval_score


def get_search_space(
        setts: settings.TaggingEvalRangesSettings) -> dict[str, list[Any]]:
    def get_mapping(name: str, selection: Selection) -> dict[str, list[Any]]:
        print(name, selection)
        if isinstance(selection, tuple):
            assert len(selection) > 1
            if isinstance(selection[0], bool):
                return {name: selection}  # type: ignore
            elif isinstance(selection[0], int):
                assert len(selection) == 2
                assert isinstance(selection[1], int)
                return {name: list(range(selection[0], selection[1]+1))}
            else:
                assert isinstance(selection[0], float)
                assert len(
                    selection) == 3, "Must specify float space including step"
                assert isinstance(selection[1], float)
                return {name: [
                    selection[0]+selection[2]*x for x in range(
                        int((selection[1]-selection[0])/selection[2])+1)]}

        else:
            assert isinstance(selection, dict)
            out_dict: dict[str, list[Any]] = {}
            for sub_name, sub_selection in selection.items():
                out_dict |= get_mapping(
                    f"{name}_{sub_name}",
                    sub_selection
                )
            return out_dict
    return {
        n: space
        for name, selection in dataclasses.asdict(setts).items()
        if selection is not None
        for n, space in get_mapping(name, selection).items()}


def eval_optimise(args: settings.EvalOptSettings):
    search_space = get_search_space(args.ranges)
    print(search_space)

    sampler_path = f"{args.opt_path}/{args.study_name}_sampler.pkl"
    if args.tagging.mode == "continue" and os.path.exists(
            sampler_path):
        sampler = pickle.load(open(sampler_path, "rb"))
    else:
        sampler = optuna.samplers.GridSampler(
            search_space=search_space,
            seed=args.tagging.seed)
    # group not needed due to no tree-structured sampling space

    storage_name = f"sqlite:///{args.opt_path}/{args.study_name}.db"
    storage = optuna.storages.RDBStorage(
        url=storage_name, engine_kwargs={"connect_args": {"timeout": 100}})
    study = optuna.create_study(
        sampler=sampler,
        study_name=args.study_name,
        direction="maximize",
        load_if_exists=args.tagging.mode == "continue",
        storage=storage,
    )

    gpu_queue = GpuQueue()
    objective = EvalObjective(
        args, gpu_queue, sampler_path)

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=len(gpu_queue.all_idxs))
