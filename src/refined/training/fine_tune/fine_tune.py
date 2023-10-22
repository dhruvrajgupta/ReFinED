import os
import shutil
import time
from statistics import mean
from typing import Dict, Iterable, Optional

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import trange, tqdm
from transformers import get_linear_schedule_with_warmup

from refined.data_types.doc_types import Doc
from refined.dataset_reading.entity_linking.document_dataset import DocDataset, DocIterDataset
from refined.evaluation.evaluation import get_datasets_obj, evaluate
from refined.inference.processor import Refined
from refined.training.fine_tune.fine_tune_args import FineTuningArgs, parse_fine_tuning_args
from refined.training.train.training_args import TrainingArgs
from refined.utilities.general_utils import get_logger
import wandb

LOG = get_logger(name=__name__)
wandb.login()

def main():
    languages = ["en", "de"]

    fine_tuning_args = parse_fine_tuning_args()
    
    # Checkpoint size same as that of eval set
    fine_tuning_args.checkpoint_every_n_steps = 10
    fine_tuning_args.checkpoint_num = 0

    # Check language present
    if fine_tuning_args.language not in languages:
        LOG.error(f"Language {fine_tuning_args.language} not available")
        return
    
    # Train DS
    train_ds = {
        "debug": f"{fine_tuning_args.language}_wikipedia_links_aligned_train_100_sample.json",
        "10p" : f"{fine_tuning_args.language}_wikipedia_links_aligned_train_10p.json"
    }

    if fine_tuning_args.ds_percent not in train_ds.keys():
        LOG.error(\
            f"Training dataset '{fine_tuning_args.language}_wikipedia_links_aligned_train_{fine_tuning_args.ds_percent}' "\
                "not found.")
        return
    
    # DIRS
    # dir_path=f"refined/offline_data_generation/data/organised_data_dir_{fine_tuning_args.language}"

    dir_path = f"../../offline_data_generation/data/organised_data_dir_{fine_tuning_args.language}"
    train_path = f"{dir_path}/datasets/{train_ds[fine_tuning_args.ds_percent]}"
    eval_path = f"{dir_path}/datasets/{fine_tuning_args.language}_wikipedia_links_aligned_eval_20.json"
    # eval_path = f"{dir_path}/datasets/{fine_tuning_args.language}_wikipedia_links_aligned_eval_1e4.json"

    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    LOG.info("Fine-tuning end-to-end EL" if fine_tuning_args.el else "Fine-tuning ED only.")
    # refined = Refined.from_pretrained(model_name=fine_tuning_args.model_name,
    #                                   entity_set=fine_tuning_args.entity_set,
    #                                   use_precomputed_descriptions=fine_tuning_args.use_precomputed_descriptions,
    #                                   device=fine_tuning_args.device)

    run = wandb.init(
        # Set the project where this run will be logged
        project="multilingual-refined",
        # Track hyperparameters and run metadata
        config=fine_tuning_args
    )

    wandb.define_metric("checkpoint_num")
    wandb.define_metric("epoch")
    wandb.define_metric("EL/*", step_metric="checkpoint_num")
    wandb.define_metric("ED/*", step_metric="checkpoint_num")
    wandb.define_metric("EL_average_f1", step_metric="checkpoint_num")
    wandb.define_metric("ED_average_f1", step_metric="checkpoint_num")
    wandb.define_metric("LR/*", step_metric="epoch")

    refined = Refined.from_pretrained(model_name="wikipedia_model",
                                    #   data_dir="refined/offline_data_generation/data/organised_data_dir",
                                      data_dir=dir_path,
                                      entity_set=fine_tuning_args.entity_set,
                                      use_precomputed_descriptions=fine_tuning_args.use_precomputed_descriptions,
                                      device=fine_tuning_args.device, 
                                      debug=True,
                                      download_files=False,
                                      lang=fine_tuning_args.language)

    datasets = get_datasets_obj(preprocessor=refined.preprocessor)

    de_train_dataset_name_to_docs = {
        "DE_WIKI_TRAIN": datasets.get_generic_lang_docs(
            # filename="refined/offline_data_generation/data/organised_data_dir/datasets/wikipedia_links_aligned_spans_dev_8.json",
            # filename="refined/offline_data_generation/data/organised_data_dir/datasets/wikipedia_links_aligned_train.json",
            filename=train_path,
            include_gold_label=True,
            filter_not_in_kb=True,
            include_spans=True
        )
    }

    LOG.info("Loading eval on mem...")

    de_eval_dataset_name_to_docs = {
        "DE_WIKI_EVAL": list(datasets.get_generic_lang_docs(
            # filename="refined/offline_data_generation/data/organised_data_dir/datasets/wikipedia_links_aligned_spans_eval_2.json",
            # filename="refined/offline_data_generation/data/organised_data_dir/datasets/wikipedia_links_aligned_eval_10.json",
            # filename="refined/offline_data_generation/data/organised_data_dir/datasets/wikipedia_links_aligned_eval.json",
            filename=eval_path,
            include_gold_label=True,
            filter_not_in_kb=True,
            include_spans=True
        ))
    }

    # evaluation_dataset_name_to_docs = {
    #     "AIDA": list(datasets.get_aida_docs(
    #         split="dev",
    #         include_gold_label=True,
    #         filter_not_in_kb=True,
    #         include_spans=True,
    #     ))
    # }

    # start_fine_tuning_task(refined=refined,
    #                        fine_tuning_args=fine_tuning_args,
    #                        train_docs=list(datasets.get_aida_docs(split="train", include_gold_label=True)),
    #                        evaluation_dataset_name_to_docs=evaluation_dataset_name_to_docs)

    start_fine_tuning_task(refined=refined,
                           fine_tuning_args=fine_tuning_args,
                           train_docs=de_train_dataset_name_to_docs["DE_WIKI_TRAIN"],
                           evaluation_dataset_name_to_docs=de_eval_dataset_name_to_docs)


def start_fine_tuning_task(refined: 'Refined', train_docs: Iterable[Doc],
                           evaluation_dataset_name_to_docs: Dict[str, Iterable[Doc]],
                           fine_tuning_args: FineTuningArgs):

    total_ds_length = 100000

    ds_length = {
        "debug": 100,
        "10p" : int(total_ds_length * 0.10)
    }

    LOG.info("Fine-tuning end-to-end EL" if fine_tuning_args.el else "Fine-tuning ED only.")
    train_docs = train_docs
    training_dataset = DocIterDataset(
        docs=train_docs,
        preprocessor=refined.preprocessor,
        # approximate_length=8
        approximate_length=ds_length[fine_tuning_args.ds_percent]
    )
    training_dataloader = DataLoader(
        dataset=training_dataset, batch_size=fine_tuning_args.batch_size, num_workers=1,
        collate_fn=training_dataset.collate
    )

    model = refined.model

    if fine_tuning_args.restore_model_path is not None:
        LOG.info(f'Restored model from {fine_tuning_args.restore_model_path}')
        checkpoint = torch.load(fine_tuning_args.restore_model_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)

    for params in model.parameters():
        params.requires_grad = True

    model.entity_disambiguation.dropout.p = fine_tuning_args.ed_dropout
    model.entity_typing.dropout.p = fine_tuning_args.et_dropout

    param_groups = [
        {"params": model.get_et_params(), "lr": fine_tuning_args.lr * 100},
        {"params": model.get_desc_params(), "lr": fine_tuning_args.lr},
        {"params": model.get_ed_params(), "lr": fine_tuning_args.lr * 100},
        {"params": model.get_parameters_not_to_scale(), "lr": fine_tuning_args.lr}
    ]
    if fine_tuning_args.el:
        param_groups.append({"params": model.get_md_params(), "lr": fine_tuning_args.lr})

    optimizer = AdamW(param_groups, lr=fine_tuning_args.lr, eps=1e-8)

    total_steps = len(training_dataloader) * fine_tuning_args.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=fine_tuning_args.num_warmup_steps,
        num_training_steps=total_steps / fine_tuning_args.gradient_accumulation_steps
    )

    run_fine_tuning_loops(refined=refined, fine_tuning_args=fine_tuning_args,
                          training_dataloader=training_dataloader, optimizer=optimizer,
                          scheduler=scheduler, evaluation_dataset_name_to_docs=evaluation_dataset_name_to_docs,
                          checkpoint_every_n_steps=fine_tuning_args.checkpoint_every_n_steps)


def run_fine_tuning_loops(refined: Refined, fine_tuning_args: TrainingArgs, training_dataloader: DataLoader,
                          optimizer: AdamW, scheduler, evaluation_dataset_name_to_docs: Dict[str, Iterable[Doc]],
                          checkpoint_every_n_steps: int = 1000000, scaler: GradScaler = GradScaler()):
    model = refined.model
    best_f1 = 0.0
    for epoch_num in trange(fine_tuning_args.epochs):
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        model.train()
        LOG.info(f"Starting epoch number {epoch_num}")
        wandb.log({"epoch": epoch_num})
        for param_group in optimizer.param_groups:
            LOG.info(f"lr: {param_group['lr']}")
        wandb.log({
            "LR/et_lr": optimizer.param_groups[0]["lr"],
            "LR/desc_lr": optimizer.param_groups[1]["lr"],
            "LR/ed_lr": optimizer.param_groups[2]["lr"],
            "LR/params_not_to_scale_lr": optimizer.param_groups[3]["lr"],
            "LR/md_lr": optimizer.param_groups[4]["lr"],
            })
        total_loss = 0.0
        for step, batch in tqdm(enumerate(training_dataloader), total=len(training_dataloader)):
            batch = batch.to(fine_tuning_args.device)
            with autocast():
                output = model(batch=batch)
                loss = output.ed_loss + output.et_loss + (output.description_loss * 0.01)
                if fine_tuning_args.el:
                    loss += output.md_loss * 0.01
                if fine_tuning_args.gradient_accumulation_steps >= 1:
                    loss = loss / fine_tuning_args.gradient_accumulation_steps

            loss = loss.mean()
            total_loss += loss.item()

            if step % 100 == 99 or step == 1:
                LOG.info(f"Loss: {total_loss / step}")
                wandb.log({"Loss": (total_loss / step)})

            scaler.scale(loss).backward()

            if (step + 1) % fine_tuning_args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            if (step + 1) % checkpoint_every_n_steps == 0:
                best_f1 = run_checkpoint_eval_and_save(best_f1, evaluation_dataset_name_to_docs, fine_tuning_args,
                                                       refined, optimizer=optimizer, scaler=scaler,
                                                       scheduler=scheduler, step_num=step+1, epoch_num=epoch_num)

        best_f1 = run_checkpoint_eval_and_save(best_f1, evaluation_dataset_name_to_docs, fine_tuning_args,
                                               refined, optimizer=optimizer, scaler=scaler,
                                               scheduler=scheduler, step_num=step+1, epoch_num=epoch_num)


def run_checkpoint_eval_and_save(best_f1: float, evaluation_dataset_name_to_docs: Dict[str, Iterable[Doc]],
                                 fine_tuning_args: TrainingArgs, refined: Refined, optimizer: AdamW,
                                 scaler: GradScaler,
                                 scheduler, 
                                 # Used for logging
                                 step_num, epoch_num):
    fine_tuning_args.checkpoint_num += 1
    checkpoint_num = fine_tuning_args.checkpoint_num
    wandb.log({"checkpoint_num": checkpoint_num})

    torch.cuda.empty_cache()
    evaluation_metrics = evaluate(refined=refined,
                                  evaluation_dataset_name_to_docs=evaluation_dataset_name_to_docs,
                                  el=fine_tuning_args.el,  # only evaluate EL when training EL
                                  ed=True,  # always evaluate standalone ED
                                  ed_threshold=fine_tuning_args.ed_threshold,
                                  step_num=step_num,
                                  epoch_num=epoch_num,
                                  checkpoint_num=checkpoint_num)
    
    for metrics in evaluation_metrics.values():
        if metrics.el:
            el_metrics = metrics
            wandb.log({
                "EL/precision": el_metrics.get_precision(), 
                "EL/recall": el_metrics.get_recall(), 
                "EL/f1": el_metrics.get_f1(), 
                "EL/accuracy": el_metrics.get_accuracy(), 
                "EL/gold_recall": el_metrics.get_gold_recall(),
            })
            # MD results only make sense for when EL mode is enabled
            wandb.log({
                "EL/MD_f1": el_metrics.get_f1_md(), 
                "EL/MD_precision": el_metrics.get_precision_md(), 
                "EL/MD_recall":el_metrics.get_recall_md()
            })
        else:
            ed_metrics = metrics
            wandb.log({
                "ED/precision": ed_metrics.get_precision(), 
                "ED/recall": ed_metrics.get_recall(), 
                "ED/f1": ed_metrics.get_f1(), 
                "ED/accuracy": ed_metrics.get_accuracy(), 
                "ED/gold_recall": ed_metrics.get_gold_recall(),
            })
    
    if fine_tuning_args.checkpoint_metric == 'el':
        LOG.info("Using EL performance for checkpoint metric")
        average_f1 = mean([metrics.get_f1() for metrics in evaluation_metrics.values() if metrics.el])
        wandb.log({"EL_average_f1": average_f1})
    elif fine_tuning_args.checkpoint_metric == 'ed':
        LOG.info("Using ED performance for checkpoint metric")
        average_f1 = mean([metrics.get_f1() for metrics in evaluation_metrics.values() if not metrics.el])
        wandb.log({"ED_average_f1": average_f1})
    else:
        raise Exception("--checkpoint_metric (`checkpoint_metric`) needs to be set to el or ed,")

    if average_f1 > best_f1:
        LOG.info(f"Obtained best F1 so far of {average_f1:.3f} (previous best {best_f1:.3f})")
        best_f1 = average_f1
        wandb.run.summary["eval_best_f1"] = f"{average_f1:.3f}"
        model_output_dir = os.path.join(fine_tuning_args.output_dir, fine_tuning_args.experiment_name)
        if os.path.exists(model_output_dir):
            shutil.rmtree(model_output_dir)
        model_output_dir = os.path.join(model_output_dir, f"f1_{average_f1:.4f}")
        os.makedirs(model_output_dir, exist_ok=True)
        LOG.info(f"Storing model at {model_output_dir} along with optimizer, scheduler, and scaler")
        model_to_save = (
            refined.model.module if hasattr(refined.model, "module") else refined.model
        )
        torch.save(model_to_save.state_dict(), os.path.join(model_output_dir, "model.pt"))
        fine_tuning_args.to_file(os.path.join(model_output_dir, "fine_tuning_args.json"))
        model_to_save.config.to_file(os.path.join(model_output_dir, "config.json"))

        # save optimiser, scheduler, and scaler so training can be resumed if it crashes
        torch.save(optimizer.state_dict(), os.path.join(model_output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(model_output_dir, "scheduler.pt"))
        torch.save(scaler.state_dict(), os.path.join(model_output_dir, "scaler.pt"))

    torch.cuda.empty_cache()
    return best_f1


def fine_tune_on_docs(refined: Refined, train_docs: Iterable[Doc], eval_docs: Iterable[Doc],
                      fine_tuning_args: Optional[FineTuningArgs] = None):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if fine_tuning_args is None:
        fine_tuning_args = FineTuningArgs(experiment_name=f'{int(time.time())}')
    evaluation_dataset_name_to_docs = {
        "custom_eval_dataset": list(eval_docs)
    }
    start_fine_tuning_task(refined=refined, train_docs=train_docs, fine_tuning_args=fine_tuning_args,
                           evaluation_dataset_name_to_docs=evaluation_dataset_name_to_docs)


if __name__ == "__main__":
    main()
