import os
import logging
import pickle

import torch
import transformers
from bitsandbytes.optim import Adam8bit
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm
import pathlib

from . import model, dataset, evaluate
from .. import extraction, data, settings

from typing import Mapping, Sequence

BERT = ("bert-base-multilingual-cased",)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def initialize_tag_system(
        ds: str,
        tag_vocab_path: pathlib.Path = pathlib.Path(".")) -> dict[str, int]:
    with open(tag_vocab_path / (ds + '.pkl'), 'rb') as f:
        tag_vocab = pickle.load(f)

    return tag_vocab


def save_vocab(args: settings.Settings):
    data_path = pathlib.Path(args.file.data_folder)
    prefix = args.file.conllu_file

    train_reader = data.load_conllu(
        prefix, "train", dir=data_path)
    _, sup2id = extraction.prepare_train(
        train_reader,
        arguments=args.deprels.arguments,
        adjuncts=args.deprels.adjuncts,
        delete=args.deprels.delete,
        merged=args.deprels.merged,
        without_labels=not args.deprels.labelled,
        merged_fallback_subtypes=args.deprels.merged_fallback_subtypes,
        distinguish_merged_fallback_subtypes=(
            args.deprels.distinguish_merged_fallback_subtypes),
        order_relations=args.deprels.order_relations,
        )

    path = pathlib.Path(args.tagging.tag_vocab_path)
    path.mkdir(parents=True, exist_ok=True)
    with (path
            / (args.file.conllu_file + '.pkl')).open("wb+", ) as f:
        pickle.dump(sup2id, f)


def prepare_training_data(
        train_data: Sequence[Sequence[tuple[str, str, str]]],
        eval_data: Sequence[Sequence[tuple[str, str, str]]],
        dataset_name: str,
        tag_system: Mapping[str, int],
        model_name: str,
        batch_size: int
        ) -> tuple[
            dataset.TaggingDataset, dataset.TaggingDataset,
            DataLoader, DataLoader]:

    tokeniser = transformers.AutoTokenizer.from_pretrained(
        model_name, truncation=True, use_fast=True)

    train_dataset = dataset.TaggingDataset(
        "train", tokeniser, tag_system, train_data, device, dataset_name)
    eval_dataset = dataset.TaggingDataset(
        "eval", tokeniser, tag_system, eval_data, device, dataset_name)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size,
        collate_fn=train_dataset.collate,
        pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, collate_fn=eval_dataset.collate,
        pin_memory=True
    )
    return train_dataset, eval_dataset, train_dataloader, eval_dataloader


def prepare_test_data(
        test_data: Sequence[Sequence[tuple[str, str, str]]],
        dataset_name: str,
        tag_system: Mapping[str, int],
        model_name: str,
        batch_size: int) -> tuple[dataset.TaggingDataset, DataLoader]:

    print(f"Evaluating {model_name}")
    tokeniser = transformers.AutoTokenizer.from_pretrained(
        model_name, truncation=True, use_fast=True)
    test_dataset = dataset.TaggingDataset(
        "test", tokeniser, tag_system, test_data, device,
        dataset_name
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=test_dataset.collate
    )
    return test_dataset, test_dataloader


def generate_config(
        model_type: str, tag_system: Mapping[str, int], model_path: str):
    if model_type in BERT:
        config = transformers.AutoConfig.from_pretrained(
            model_path,
            num_labels=len(tag_system)+1,
        )
        config.task_specific_params = {
                'model_path': model_path,
                'pos_emb_dim': 256,
                'num_pos_tags': 50,
                'lstm_layers': 3,
                'dropout': 0.33,
                'use_pos': True,
                'n_heads': 12,
                'transformer_layers': 4
            }
    else:
        logging.error("Invalid model type.")
        return
    return config


def initialize_model(
        model_type: str, tag_system: Mapping[str, int], model_path: str
        ) -> model.ModelForTagging | None:
    config = generate_config(
        model_type, tag_system, model_path
    )
    if model_type in BERT:
        m = model.ModelForTagging(config=config)
        # m = torch.compile(m)  # type: ignore
    else:
        logging.error("Invalid model type")
        return None
    return m


def initialize_optimizer_and_scheduler(
        model, dataset_size, lr=5e-5, num_epochs=4,
        num_warmup_steps=160):
    num_training_steps = num_epochs * dataset_size
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if "bert" not in n],
            "weight_decay": 0.0,
            "lr": lr * 50, "betas": (0.9, 0.9),
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       "bert" in n and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": lr, "betas": (0.9, 0.999),
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       "bert" in n and not any(nd in n for nd in no_decay)],
            "weight_decay": 0.1,
            "lr": lr, "betas": (0.9, 0.999),
        },
    ]

    optimizer = Adam8bit(
        grouped_parameters, lr=lr
    )
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    return optimizer, scheduler, num_training_steps


def register_run_metrics(
        writer, run_name, lr, epochs, tag_accuracy):
    writer.add_hparams(
        {'run_name': run_name, 'lr': lr, 'epochs': epochs},
        {'tag_accuracy': tag_accuracy})


def train_command(args: settings.Settings):
    data_path = pathlib.Path(args.file.data_folder)
    prefix: str = args.file.conllu_file

    train_reader = data.load_conllu(prefix, "train", dir=data_path)
    dev_reader = data.load_conllu(prefix, "dev", dir=data_path)
    logging.info("Preparing Data")

    train_data, sup2id = extraction.prepare_train(
        train_reader,
        arguments=args.deprels.arguments,
        adjuncts=args.deprels.adjuncts,
        delete=args.deprels.delete,
        merged=args.deprels.merged,
        without_labels=not args.deprels.labelled,
        merged_fallback_subtypes=args.deprels.merged_fallback_subtypes,
        distinguish_merged_fallback_subtypes=(
            args.deprels.distinguish_merged_fallback_subtypes),
        order_relations=args.deprels.order_relations,
        )
    dev_data = extraction.prepare(
        dev_reader,
        arguments=args.deprels.arguments,
        adjuncts=args.deprels.adjuncts,
        delete=args.deprels.delete,
        merged=args.deprels.merged,
        without_labels=not args.deprels.labelled,
        merged_fallback_subtypes=args.deprels.merged_fallback_subtypes,
        distinguish_merged_fallback_subtypes=(
            args.deprels.distinguish_merged_fallback_subtypes),
        order_relations=args.deprels.order_relations,
        )

    logging.info("Preparing Data")
    train_dataset, dev_dataset, train_dataloader, dev_dataloader = (
        prepare_training_data(
            train_data, dev_data, prefix,
            sup2id, args.tagging.model_name, args.tagging.batch_size))

    logging.info("Initializing The Model")
    model = initialize_model(
        args.tagging.model_name, sup2id, args.tagging.model_path
    )
    assert model is not None
    model.to(device)

    train_set_size = len(train_dataloader)
    optimizer, scheduler, num_training_steps = (
        initialize_optimizer_and_scheduler(
            model, train_set_size, args.tagging.lr, args.tagging.epochs,
            args.tagging.num_warmup_steps
        )
    )

    optimizer.zero_grad()
    run_name = (
        args.file.conllu_file + "-" + args.tagging.model_name + "-" + str(
            args.tagging.lr) + "-" + str(args.tagging.epochs))
    writer = None
    if args.tagging.use_tensorboard:
        writer = SummaryWriter(comment=run_name)

    logging.info("Starting The Training Loop")
    model.train()
    n_iter = 0

    last_acc: float = 0
    best_acc: float = 0
    tol = 99999

    for epo in tqdm.tqdm(range(args.tagging.epochs)):
        logging.info(f"*******************EPOCH {epo}*******************")
        t = 1
        model.train()

        with tqdm.tqdm(train_dataloader, disable=False) as progbar:
            for batch in progbar:
                batch = {k: v.to(device) for k, v in batch.items()}

                with torch.amp.autocast(
                        "cpu" if device == torch.device("cpu") else "cuda",
                        enabled=True, dtype=torch.bfloat16
                        ):
                    outputs = model(**batch)

                loss = outputs[0]
                loss.mean().backward()
                if args.tagging.use_tensorboard:
                    assert writer is not None
                    writer.add_scalar('Loss/train', torch.mean(loss), n_iter)
                progbar.set_postfix(loss=torch.mean(loss).item())

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # debug_optimizer_devices(model, optimizer)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                n_iter += 1
                t += 1

        if True:  # evaluation at the end of epoch
            predictions, eval_labels = evaluate.predict(
                model, dev_dataloader, len(dev_dataset),
                len(sup2id), args.tagging.batch_size, device
            )
            dev_acc = evaluate.calc_tag_accuracy(
                predictions, eval_labels, writer,
                args.tagging.use_tensorboard)

            if args.tagging.use_tensorboard:
                assert writer is not None
                writer.add_scalar(
                    'acc/dev',
                    dev_acc, n_iter)

            logging.info("current acc {}".format(dev_acc))
            logging.info("last acc {}".format(last_acc))
            logging.info("best acc {}".format(best_acc))

            # if dev_metrics.fscore > last_fscore or dev_loss < last...
            if dev_acc > best_acc:
                tol = 99999
                logging.info("tol refill")
                logging.info("save the best model")
                best_acc = dev_acc
                _save_best_model(
                    model, pathlib.Path(
                        args.tagging.output_path), run_name)
            else:
                tol -= 1

            if tol < 0:
                _finish_training(
                    model, sup2id, dev_dataloader,
                    dev_dataset, run_name, writer, args.tagging)
                return
            # end of epoch
            pass

    _finish_training(
        model, sup2id, dev_dataloader, dev_dataset,
        run_name, writer, args.tagging)


def _save_best_model(
        model: torch.nn.Module, output_path: pathlib.Path, run_name: str):
    logging.info("Saving The Newly Found Best Model")
    os.makedirs(output_path, exist_ok=True)
    to_save_file = os.path.join(output_path, run_name)
    torch.save(model.state_dict(), to_save_file)


def debug_optimizer_devices(model, optimizer):
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("current device:", torch.cuda.current_device())

    for name, p in model.named_parameters():
        if p.requires_grad:
            print(
                name,
                "param:", p.device,
                "grad:", None if p.grad is None else p.grad.device,
                "shape:", tuple(p.shape),
            )
            state = optimizer.state.get(p, {})
            for k, v in state.items():
                if torch.is_tensor(v):
                    print(f"  state[{k}]:", v.device, tuple(v.shape))
                else:
                    print(f"  state[{k}]:", type(v).__name__, v)
            print("-" * 60)


def _finish_training(
        model: torch.nn.Module, sup2id: Mapping[str, int],
        eval_dataloader: DataLoader,
        eval_dataset: dataset.TaggingDataset,
        run_name: str, writer: None | SummaryWriter,
        args: settings.TaggingSettings):

    predictions, eval_labels = evaluate.predict(
        model, eval_dataloader, len(eval_dataset),
        len(sup2id), args.batch_size,
        device)
    acc = evaluate.calc_tag_accuracy(
        predictions, eval_labels, writer,
        args.use_tensorboard)
    register_run_metrics(
        writer, run_name, args.lr,
        args.epochs, acc)


def decode_model_name(model_name):
    name_chunks = model_name.split("-")
    name_chunks = name_chunks[1:]
    if name_chunks[0] == "td" or name_chunks[0] == "bu":
        tagging_schema = name_chunks[0] + "-" + name_chunks[1]
        model_type = name_chunks[2]
    else:
        tagging_schema = name_chunks[0]
        model_type = name_chunks[1]
    return tagging_schema, model_type


def evaluate_command(args: settings.Settings):
    data_path: pathlib.Path = pathlib.Path(
        args.file.data_folder)

    print("Evaluation Args", args)
    prefix: str = args.file.conllu_file

    test_reader = data.load_conllu(prefix, "test", dir=data_path)
    test_data = extraction.prepare(
        test_reader,
        arguments=args.deprels.arguments,
        adjuncts=args.deprels.adjuncts,
        delete=args.deprels.delete,
        merged=args.deprels.merged,
        without_labels=not args.deprels.labelled,
        merged_fallback_subtypes=args.deprels.merged_fallback_subtypes,
        distinguish_merged_fallback_subtypes=(
            args.deprels.distinguish_merged_fallback_subtypes),
        order_relations=args.deprels.order_relations,
        )

    sup2id = initialize_tag_system(
        prefix, pathlib.Path(args.tagging.tag_vocab_path)
    )

    writer = SummaryWriter(comment=args.tagging.model_name)

    logging.info("Preparing Data")
    eval_dataset, eval_dataloader = prepare_test_data(
        test_data, prefix, sup2id, args.tagging.model_name,
        args.tagging.batch_size)

    model = initialize_model(
        args.tagging.model_name, sup2id, args.tagging.model_path)
    assert model is not None

    model.load_state_dict(
        torch.load(args.tagging.model_path + args.tagging.model_name))
    model.to(device)

    predictions, eval_labels = evaluate.predict(
        model, eval_dataloader, len(eval_dataset),
        len(sup2id), args.tagging.batch_size, device)

    dev_acc = evaluate.calc_tag_accuracy(
        predictions, eval_labels,
        writer, args.tagging.use_tensorboard)

    print(
        "acc:", dev_acc)


def predict_command(args: settings.Settings):
    data_path: pathlib.Path = pathlib.Path(
        args.file.data_folder)

    print("predict Args", args)

    prefix: str = args.file.conllu_file

    pred_reader = data.load_conllu(prefix, None, dir=data_path)
    pred_data = extraction.prepare(
        pred_reader,
        arguments=args.deprels.arguments,
        adjuncts=args.deprels.adjuncts,
        delete=args.deprels.delete,
        merged=args.deprels.merged,
        without_labels=not args.deprels.labelled,
        merged_fallback_subtypes=args.deprels.merged_fallback_subtypes,
        distinguish_merged_fallback_subtypes=(
            args.deprels.distinguish_merged_fallback_subtypes),
        order_relations=args.deprels.order_relations,
        )

    logging.info("Initializing Tag System")
    sup2id = initialize_tag_system(
        prefix, pathlib.Path(args.tagging.tag_vocab_path)
    )

    logging.info("Preparing Data")
    pred_dataset, pred_dataloader = prepare_test_data(
        pred_data, prefix, sup2id, args.tagging.model_name,
        args.tagging.batch_size)

    model = initialize_model(
        args.tagging.model_name, sup2id, args.tagging.model_path)
    assert model is not None

    model.load_state_dict(torch.load(
        args.tagging.model_path + args.tagging.model_name))
    model.to(device)

    predictions, eval_labels = evaluate.predict(
        model, pred_dataloader, len(pred_dataset),
        len(sup2id), args.tagging.batch_size, device)

    id2sup = {i: sup for sup, i in sup2id.items()}
    pred_ids = predictions[eval_labels != -1].argmax(-1)

    supertags = [id2sup[i] for i in pred_ids]

    with open(
            args.tagging.output_path
            + args.tagging.model_name
            + ".pred.json", "w") as fout:
        print(
            "Saving predictions to",
            args.tagging.output_path + args.tagging.model_name
            + ".pred.json")
        for sup in supertags:
            fout.write(sup)
