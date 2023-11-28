import os
from typing import List

import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from refined.data_types.doc_types import Doc
from refined.dataset_reading.entity_linking.wikipedia_dataset import WikipediaDataset
from refined.dataset_reading.entity_linking.document_dataset import ShuffleDataset
from refined.doc_preprocessing.preprocessor import PreprocessorInferenceOnly
from refined.doc_preprocessing.wikidata_mapper import WikidataMapper
from refined.inference.processor import Refined
from refined.model_components.config import NER_TAG_TO_IX, ModelConfig
from refined.model_components.refined_model import RefinedModel
from refined.resource_management.aws import S3Manager
from refined.resource_management.resource_manager import ResourceManager
from refined.torch_overrides.data_parallel_refined import DataParallelReFinED
from refined.training.fine_tune.fine_tune import run_fine_tuning_loops
from refined.training.train.training_args import parse_training_args
from refined.utilities.general_utils import get_logger
import wandb
import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data")

LOG = get_logger(name=__name__)


def main():
    wandb.login()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # DDP (ensure batch_elements_included is used)

    training_args = parse_training_args()
    training_args.checkpoint_num = 0

    languages = ["en", "de", "es", "pt", "ru"]

    # Check language present
    if training_args.language not in languages:
        LOG.error(f"Language {training_args.language} not available")
        return

    # Train DS
    train_ds = {
        # "debug": f"{training_args.language}_wikipedia_links_aligned_spans_train_100_sample.json",
        "debug": f"{training_args.language}_wikipedia_links_aligned_spans_100_sample.json",
        "10p" : f"{training_args.language}_wikipedia_links_aligned_spans_train_10p.json",
        "20p" : f"{training_args.language}_wikipedia_links_aligned_spans_train_20p.json",
        "30p" : f"{training_args.language}_wikipedia_links_aligned_spans_train_30p.json",
        "40p" : f"{training_args.language}_wikipedia_links_aligned_spans_train_40p.json",
        "50p" : f"{training_args.language}_wikipedia_links_aligned_spans_train_50p.json",
        "60p" : f"{training_args.language}_wikipedia_links_aligned_spans_train_60p.json",
        "70p" : f"{training_args.language}_wikipedia_links_aligned_spans_train_70p.json",
        "80p" : f"{training_args.language}_wikipedia_links_aligned_spans_train_80p.json",
        "90p" : f"{training_args.language}_wikipedia_links_aligned_spans_train_90p.json",
        "100p" : f"{training_args.language}_wikipedia_links_aligned_spans_train_100p.json"
    }

    if training_args.ds_percent not in train_ds.keys():
        LOG.error(\
            f"Training dataset '{training_args.language}_wikipedia_links_aligned_train_{training_args.ds_percent}' "\
                "not found.")
        return

    # DIRS
    # dir_path=f"refined/offline_data_generation/data/organised_data_dir_{training_args.language}"

    dir_path = f"../../offline_data_generation/data/organised_data_dir_{training_args.language}"
    train_path = f"{dir_path}/datasets/{train_ds[training_args.ds_percent]}"
    if training_args.ds_percent == "debug":
        eval_path = f"{dir_path}/datasets/{training_args.language}_wikipedia_links_aligned_spans_20_sample.json"
        # eval_path = f"{dir_path}/datasets/{training_args.language}_wikipedia_links_aligned_spans_val_1e4_conf.json"
    else:
        eval_path = f"{dir_path}/datasets/{training_args.language}_wikipedia_links_aligned_spans_val_1e4_conf.json"
    # eval_path = f"{dir_path}/datasets/wikipedia_links_aligned.json_spans_small.json"
    # eval_path = f"{dir_path}/datasets/{training_args.language}_wikipedia_links_aligned_spans_eval_20.json"
    # eval_path = f"{dir_path}/datasets/{training_args.language}_wikipedia_links_aligned_eval_1e4.json"

    training_args.download_files = False
    training_args.data_dir = dir_path
    training_args.stepp = 0

    if training_args.ds_percent == "debug":
        training_args.checkpoint_every_n_steps = 5
        # Resuming training
        training_args.epochs = 10000
        training_args.debug = True

    if training_args.resume:
        model_output_dir = os.path.join(training_args.output_dir, training_args.experiment_name)
        model_output_dir = f"{model_output_dir}-epoch_checkpoint"
        LOG.info(f'Restored log state from {model_output_dir}')
        with open(f'{model_output_dir}/log_state.json', 'r') as file:
            log_state = json.load(file)
            LOG.info(log_state)
            training_args.checkpoint_num = log_state["checkpoint_num"]
            training_args.start_epoch = log_state["epoch"]+1
            training_args.stepp = log_state["stepp"]

    total_ds_length = 100000

    ds_length = {
        "debug": 100,
        "10p" : int(total_ds_length * 0.10),
        "20p" : int(total_ds_length * 0.20),
        "30p" : int(total_ds_length * 0.30),
        "40p" : int(total_ds_length * 0.40),
        "50p" : int(total_ds_length * 0.50),
        "60p" : int(total_ds_length * 0.60),
        "70p" : int(total_ds_length * 0.70),
        "80p" : int(total_ds_length * 0.80),
        "90p" : int(total_ds_length * 0.90),
        "100p" : int(total_ds_length * 1.00),
    }

    run = wandb.init(
        # Set the project where this run will be logged
        project="multilingual-refined",
        # Track hyperparameters and run metadata
        config=training_args
    )

    wandb.define_metric("checkpoint_num")
    wandb.define_metric("epoch")
    wandb.define_metric("stepp")
    wandb.define_metric("EL/*", step_metric="checkpoint_num")
    wandb.define_metric("ED/*", step_metric="checkpoint_num")
    wandb.define_metric("EL_average_f1", step_metric="checkpoint_num")
    wandb.define_metric("ED_average_f1", step_metric="checkpoint_num")
    wandb.define_metric("LR/*", step_metric="epoch")
    wandb.define_metric("Loss", step_metric="stepp")
    wandb.define_metric("epoch", step_metric="stepp")
    wandb.define_metric("checkpoint_num", step_metric="stepp")

    resource_manager = ResourceManager(S3Manager(),
                                       data_dir=training_args.data_dir,
                                       entity_set=training_args.entity_set,
                                       load_qcode_to_title=True,
                                       load_descriptions_tns=True,
                                       model_name=None
                                       )
    if training_args.download_files:
        resource_manager.download_data_if_needed()
        resource_manager.download_additional_files_if_needed()
        resource_manager.download_training_files_if_needed()

    preprocessor = PreprocessorInferenceOnly(
        data_dir=training_args.data_dir,
        debug=training_args.debug,
        max_candidates=training_args.num_candidates_train,
        transformer_name=training_args.transformer_name,
        ner_tag_to_ix=NER_TAG_TO_IX,  # for now include default ner_to_tag_ix can make configurable in future
        entity_set=training_args.entity_set,
        use_precomputed_description_embeddings=False,
        lang = training_args.language
    )

    wikidata_mapper = WikidataMapper(resource_manager=resource_manager)

    wikipedia_dataset_file_path = resource_manager.get_training_data_files()['wikipedia_training_dataset'].split("/")
    wikipedia_dataset_file_path[-1] = train_ds[training_args.ds_percent]
    wikipedia_dataset_file_path = "/".join(wikipedia_dataset_file_path)
    training_dataset = WikipediaDataset(
        # start=100,
        start=0,
        end=100000000,  # large number means every line will be read until the end of the file
        preprocessor=preprocessor,
        resource_manager=resource_manager,
        wikidata_mapper=wikidata_mapper,
        dataset_path=wikipedia_dataset_file_path,
        batch_size=training_args.batch_size,
        num_workers=8 * training_args.n_gpu,
        prefetch=100,  # add random number for each worker and have more than 2 workers to remove waiting
        mask=training_args.mask_prob,
        random_mask=training_args.mask_random_prob,
        lower_case_prob=0.05,
        candidate_dropout=training_args.candidate_dropout,
        max_mentions=training_args.max_mentions,
        sample_k_candidates=5,
        add_main_entity=True,
        file_line_count=ds_length[training_args.ds_percent]
    )
    training_dataloader = DataLoader(ShuffleDataset(training_dataset, 100, len(training_dataset), seed=training_args.seed), 
                                     batch_size=None, num_workers=8 * training_args.n_gpu,
                                     # pin_memory=True if training_args.n_gpu == 1 else False,
                                     pin_memory=True,  # may break ddp and dp training
                                     prefetch_factor=5,  # num_workers * prefetch_factor
                                     persistent_workers=True  # persistent_workers means memory is stable across epochs
                                     )
    eval_docs: List[Doc] = list(iter(WikipediaDataset(
        start=0,
        end=100000000,  # first 100 docs are used for eval
        preprocessor=preprocessor,
        resource_manager=resource_manager,
        wikidata_mapper=wikidata_mapper,
        dataset_path=eval_path,
        return_docs=True,  # this means the dataset will return `Doc` objects instead of BatchedElementsTns
        batch_size=1 * training_args.n_gpu,
        num_workers=1,
        prefetch=1,
        mask=0.0,
        random_mask=0.0,
        lower_case_prob=0.0,
        candidate_dropout=0.0,
        max_mentions=25,  # prevents memory issues
        add_main_entity=True, # add weak labels,
        file_line_count=10000
    )))

    model = RefinedModel(
        ModelConfig(data_dir=preprocessor.data_dir,
                    transformer_name=preprocessor.transformer_name,
                    ner_tag_to_ix=preprocessor.ner_tag_to_ix
                    ),
        preprocessor=preprocessor
    )

    if training_args.restore_model_path is not None:
        # TODO load `ModelConfig` file (from the directory) and initialise RefinedModel from that
        # to avoid issues when model config differs
        LOG.info(f'Restored model from {training_args.restore_model_path}')
        checkpoint = torch.load(training_args.restore_model_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)

    if training_args.resume:
        model_output_dir = os.path.join(training_args.output_dir, training_args.experiment_name)
        model_output_dir = f"{model_output_dir}-epoch_checkpoint"
        LOG.info(f'Restored model from {model_output_dir}')
        checkpoint = torch.load(f"{model_output_dir}/model.pt", map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)

    if training_args.n_gpu > 1:
        model = DataParallelReFinED(model, device_ids=list(range(training_args.n_gpu)), output_device=training_args.device)
    model = model.to(training_args.device)

    # wrap a ReFinED processor around the model so evaluation methods can be run easily
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )
    refined = Refined(
        model_file_or_model=model,
        model_config_file_or_model_config=model_to_save.config,
        preprocessor=preprocessor,
        device=training_args.device
    )

    param_groups = [
        {"params": model_to_save.get_et_params(), "lr": training_args.lr * 100},
        {"params": model_to_save.get_desc_params(), "lr": training_args.lr},
        {"params": model_to_save.get_ed_params(), "lr": training_args.lr * 100},
        {"params": model_to_save.get_parameters_not_to_scale(), "lr": training_args.lr}
    ]
    if training_args.el:
        param_groups.append({"params": model_to_save.get_md_params(), "lr": training_args.lr})

    optimizer = AdamW(param_groups, lr=training_args.lr, eps=1e-8)

    total_steps = len(training_dataloader) * training_args.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=training_args.num_warmup_steps,
        num_training_steps=total_steps / training_args.gradient_accumulation_steps
    )

    scaler = GradScaler()

    if training_args.restore_model_path is not None and training_args.resume:
        LOG.info("Restoring optimizer and scheduler")
        optimizer_checkpoint = torch.load(
            os.path.join(os.path.dirname(training_args.restore_model_path), "optimizer.pt"),
            map_location="cpu",
        )
        scheduler_checkpoint = torch.load(
            os.path.join(os.path.dirname(training_args.restore_model_path), "scheduler.pt"),
            map_location="cpu",
        )
        scaler_checkpoint = torch.load(
            os.path.join(os.path.dirname(training_args.restore_model_path), "scaler.pt"),
            map_location="cpu",
        )
        optimizer.load_state_dict(optimizer_checkpoint)
        scheduler.load_state_dict(scheduler_checkpoint)
        scaler.load_state_dict(scaler_checkpoint)

    if training_args.resume:
        model_output_dir = os.path.join(training_args.output_dir, training_args.experiment_name)
        model_output_dir = f"{model_output_dir}-epoch_checkpoint"
        LOG.info(f'Restored optimizer from {model_output_dir}')
        checkpoint = torch.load(f"{model_output_dir}/optimizer.pt", map_location='cpu')
        optimizer.load_state_dict(checkpoint)
        LOG.info(f'Restored scheduler from {model_output_dir}')
        checkpoint = torch.load(f"{model_output_dir}/scheduler.pt", map_location='cpu')
        scheduler.load_state_dict(checkpoint)
        LOG.info(f'Restored scaler from {model_output_dir}')
        checkpoint = torch.load(f"{model_output_dir}/scaler.pt", map_location='cpu')
        scaler = GradScaler()
        scaler.load_state_dict(checkpoint)
    else:
        scaler = GradScaler()

    run_fine_tuning_loops(
        refined=refined,
        fine_tuning_args=training_args,
        training_dataloader=training_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        evaluation_dataset_name_to_docs={'WIKI_DEV': eval_docs},
        checkpoint_every_n_steps=training_args.checkpoint_every_n_steps,
        scaler=scaler
    )


if __name__ == "__main__":
    main()
