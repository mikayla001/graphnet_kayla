"""Train DynEdge model for NuEBar/Tau classification with W&B logging."""

import os
from typing import List, Optional, Dict, Any

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from graphnet.data.dataset.dataset import EnsembleDataset
from graphnet.data.dataloader import DataLoader
from graphnet.data.dataset import Dataset
from graphnet.models import StandardModel
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.config import (
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
)
from graphnet.utilities.logging import Logger


def main(
    dataset_config_path: str,
    model_config_path: str,
    gpus: Optional[List[int]],
    max_epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    num_workers: int,
    suffix: Optional[str] = None,
    wandb: bool = False,
    wandb_project: str = "nuebar-tau-classification",
    wandb_entity: Optional[str] = None,
) -> None:
    """Train model from config files with W&B logging."""
    logger = Logger()

    # Initialize Weights & Biases logger
    wandb_logger = None
    if wandb:
        wandb_dir = "./wandb/"
        os.makedirs(wandb_dir, exist_ok=True)
        wandb_logger = WandbLogger(
            project=wandb_project,
            entity=wandb_entity,
            save_dir=wandb_dir,
            log_model=True,  # Save model checkpoints to W&B
            name=f"nuebar_tau_{suffix}" if suffix else "nuebar_tau_classification",
        )

    # Load configs
    model_config = ModelConfig.load(model_config_path)
    dataset_config = DatasetConfig.load(dataset_config_path)
    
    # Build model from config
    model: StandardModel = StandardModel.from_config(model_config, trust=True)
    
    # Configure training
    config = TrainingConfig(
        target=[
            target for task in model._tasks for target in task._target_labels
        ],
        early_stopping_patience=early_stopping_patience,
        fit={
            "gpus": gpus,
            "max_epochs": max_epochs,
        },
        dataloader={"batch_size": batch_size, "num_workers": num_workers},
    )
    
    # Create datasets
    datasets: Dict[str, Any] = Dataset.from_config(dataset_config)
    
    # Construct ensemble datasets for multiple selections (train_nuebar, train_tau, etc.)
    train_dataset = EnsembleDataset(
        [datasets[key] for key in datasets if key.startswith("train")]
    )
    valid_dataset = EnsembleDataset(
        [datasets[key] for key in datasets if key.startswith("validation")]
    )
    test_dataset = EnsembleDataset(
        [datasets[key] for key in datasets if key.startswith("test")]
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, **config.dataloader
    )
    valid_dataloader = DataLoader(
        valid_dataset, shuffle=False, **config.dataloader
    )
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, **config.dataloader
    )
    
    # Log configurations to W&B
    # Only log on rank-zero process for multi-GPU training
    if wandb and rank_zero_only.rank == 0:
        wandb_logger.experiment.config.update(config.as_dict())
        wandb_logger.experiment.config.update(model_config.as_dict())
        wandb_logger.experiment.config.update(dataset_config.as_dict())
        logger.info("W&B logging initialized")
    
    logger.info(f"Training on targets: {config.target}")
    logger.info(f"Train samples: {len(train_dataset)}, "
                f"Validation samples: {len(valid_dataset)}, "
                f"Test samples: {len(test_dataset)}")
    
    # Train model
    model.fit(
        train_dataloader,
        valid_dataloader,
        early_stopping_patience=config.early_stopping_patience,
        logger=wandb_logger,
        **config.fit,
    )
    
    # Save results locally
    db_name = dataset_config.path[0].split("/")[-1].split(".")[0]
    run_name = f"nuebar_tau_{suffix}" if suffix else "nuebar_tau_classification"
    output_dir = f"./output/{db_name}/{run_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving results to {output_dir}")
    model.save_state_dict(f"{output_dir}/state_dict.pth")
    
    # Get predictions and log to W&B
    if isinstance(config.target, str):
        additional_attributes = [config.target]
    else:
        additional_attributes = config.target
    
    logger.info(f"Getting predictions for: {model.prediction_labels}")
    results = model.predict_as_dataframe(
        test_dataloader,
        additional_attributes=additional_attributes + ["event_no"],
        gpus=config.fit["gpus"],
    )
    results.to_csv(f"{output_dir}/results.csv")
    
    # Log results to W&B
    if wandb and rank_zero_only.rank == 0:
        # Log results dataframe as table
        wandb_logger.experiment.log({
            "results_table": wandb.Table(dataframe=results)
        })
        # Log artifact (results CSV)
        wandb_logger.experiment.log_artifact(f"{output_dir}/results.csv")
        logger.info("Results logged to W&B")


if __name__ == "__main__":
    parser = ArgumentParser(description="Train NuEBar/Tau Classifier with W&B logging")
    
    parser.with_standard_arguments(
        "dataset-config",
        "model-config",
        "gpus",
        ("max-epochs", 50),
        "early-stopping-patience",
        ("batch-size", 32),
        ("num-workers", 4),
    )
    
    parser.add_argument(
        "--suffix",
        type=str,
        help="Suffix for output directory and W&B run name",
        default=None,
    )
    
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="W&B project name",
        default="nuebar-tau-classification",
    )
    
    parser.add_argument(
        "--wandb-entity",
        type=str,
        help="W&B entity (username or team name)",
        default=None,
    )
    
    args, _ = parser.parse_known_args()
    
    main(
        args.dataset_config,
        args.model_config,
        args.gpus,
        args.max_epochs,
        args.early_stopping_patience,
        args.batch_size,
        args.num_workers,
        args.suffix,
        args.wandb,
        args.wandb_project,
        args.wandb_entity,
    )    early_stopping_patience: int,
    batch_size: int,
    num_workers: int,
    suffix: Optional[str] = None,
    use_wandb: bool = False,  # renamed from `wandb` to avoid shadowing the module
    class_names: Optional[List[str]] = None,
) -> None:
    """Run example."""
    # Construct Logger
    logger = Logger()

    # Initialise Weights & Biases (W&B) run
    if use_wandb:
        # Make sure W&B output directory exists
        wandb_dir = "./wandb/"
        os.makedirs(wandb_dir, exist_ok=True)
        wandb_logger = WandbLogger(
            project="example-script-train_multiclassifier",
            entity="mtate9-arizona-state-university",
            save_dir=wandb_dir,
            log_model=True,
        )

    # Build model
    model_config = ModelConfig.load(model_config_path)
    model: StandardModel = StandardModel.from_config(model_config, trust=True)

    # Configuration
    config = TrainingConfig(
        target=[
            target for task in model._tasks for target in task._target_labels
        ],
        early_stopping_patience=early_stopping_patience,
        fit={
            "gpus": gpus,
            "max_epochs": max_epochs,
        },
        dataloader={"batch_size": batch_size, "num_workers": num_workers},
    )

    if suffix is not None:
        archive = os.path.join(EXAMPLE_OUTPUT_DIR, f"train_model_{suffix}")
    else:
        archive = os.path.join(EXAMPLE_OUTPUT_DIR, "train_model")
    run_name = "dynedge_{}_example".format("_".join(config.target))

    # Construct dataloaders
    dataset_config = DatasetConfig.load(dataset_config_path)
    datasets: Dict[str, Any] = Dataset.from_config(
        dataset_config,
    )

    # Construct datasets from multiple selections
    train_dataset = EnsembleDataset(
        [datasets[key] for key in datasets if key.startswith("train")]
    )
    valid_dataset = EnsembleDataset(
        [datasets[key] for key in datasets if key.startswith("valid")]
    )
    test_dataset = EnsembleDataset(
        [datasets[key] for key in datasets if key.startswith("test")]
    )

    # Construct dataloaders
    train_dataloaders = DataLoader(
        train_dataset, shuffle=True, **config.dataloader
    )
    valid_dataloaders = DataLoader(
        valid_dataset, shuffle=False, **config.dataloader
    )
    test_dataloaders = DataLoader(
        test_dataset, shuffle=False, **config.dataloader
    )

    # Log configurations to W&B
    # NB: Only log to W&B on the rank-zero process in case of multi-GPU
    #     training.
    # NOTE: the original condition here was `rank_zero_only == 0`, which
    # compares the rank_zero_only decorator object itself to an int and is
    # always False -- this block never actually ran. Fixed below.
    if use_wandb and rank_zero_only.rank == 0:
        wandb_logger.experiment.config.update(config)
        wandb_logger.experiment.config.update(model_config.as_dict())
        wandb_logger.experiment.config.update(dataset_config.as_dict())

    # Training model
    model.fit(
        train_dataloaders,
        valid_dataloaders,
        early_stopping_patience=config.early_stopping_patience,
        logger=wandb_logger if use_wandb else None,
        **config.fit,
    )

    # Save model to file
    db_name = dataset_config.path.split("/")[-1].split(".")[0]
    path = os.path.join(archive, db_name, run_name)
    os.makedirs(path, exist_ok=True)
    logger.info(f"Writing results to {path}")
    model.save_state_dict(f"{path}/state_dict.pth")

    # Get predictions
    if isinstance(config.target, str):
        additional_attributes = [config.target]
    else:
        additional_attributes = config.target

    logger.info(f"config.target: {config.target}")
    logger.info(f"prediction_columns: {model.prediction_labels}")

    results = model.predict_as_dataframe(
        test_dataloaders,
        additional_attributes=additional_attributes + ["event_no"],
        gpus=config.fit["gpus"],
    )
    results.to_csv(f"{path}/results.csv")

    # ------------------------------------------------------------------
    # Log predictions to W&B as a Table, plus a confusion matrix if this
    # is a classification task (i.e. predictions look like class scores).
    # ------------------------------------------------------------------
    if use_wandb and rank_zero_only.rank == 0:
        table = wandb.Table(dataframe=results)
        wandb.log({"test_predictions": table})

        # Attempt a confusion matrix. `model.prediction_labels` gives the
        # predicted-class column names (e.g. ['nuebar', 'tau']); the true
        # class is the target column (e.g. 'initial_state_type'), which
        # needs mapping from PDG code -> class index if class_names given.
        pred_cols = model.prediction_labels
        target_col = additional_attributes[0]

        if class_names is not None and all(
            c in results.columns for c in pred_cols
        ):
            import numpy as np

            pred_class_idx = results[pred_cols].to_numpy().argmax(axis=1)

            # Map raw target values (e.g. PDG codes) to class indices using
            # the same order as class_names. This assumes the caller passes
            # class_names in an order consistent with how the model config's
            # loss function options were defined -- verify this mapping
            # against your model config before trusting the plot.
            unique_targets = sorted(results[target_col].unique())
            if len(unique_targets) == len(class_names):
                target_to_idx = {
                    val: idx for idx, val in enumerate(unique_targets)
                }
                true_class_idx = (
                    results[target_col].map(target_to_idx).to_numpy()
                )

                wandb.log(
                    {
                        "confusion_matrix": wandb.plot.confusion_matrix(
                            y_true=true_class_idx,
                            preds=pred_class_idx,
                            class_names=class_names,
                        )
                    }
                )
            else:
                logger.warning(
                    "Number of unique target values "
                    f"({len(unique_targets)}) does not match "
                    f"len(class_names) ({len(class_names)}); skipping "
                    "confusion matrix. Check --class-names against your "
                    "actual target distribution."
                )


if __name__ == "__main__":
    # Parse command-line arguments
    parser = ArgumentParser(description="""
            Train GNN classification model.
            """)

    parser.with_standard_arguments(
        (
            "dataset-config",
            os.path.join(
                DATASETS_CONFIG_DIR,
                "training_classification_example_data_sqlite.yml",
            ),
        ),
        (
            "model-config",
            os.path.join(
                MODEL_CONFIG_DIR, "dynedge_PID_classification_example.yml"
            ),
        ),
        "gpus",
        ("max-epochs", 1),
        "early-stopping-patience",
        ("batch-size", 16),
        ("num-workers", 2),
    )

    parser.add_argument(
        "--suffix",
        type=str,
        help="Name addition to folder (default: %(default)s)",
        default=None,
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        dest="use_wandb",
        help="If True, Weights & Biases are used to track the experiment.",
    )

    parser.add_argument(
        "--class-names",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Ordered list of class names matching your model config's "
            "prediction_labels, e.g. --class-names nuebar tau. Used to "
            "build the W&B confusion matrix."
        ),
    )

    args, unknown = parser.parse_known_args()

    main(
        args.dataset_config,
        args.model_config,
        args.gpus,
        args.max_epochs,
        args.early_stopping_patience,
        args.batch_size,
        args.num_workers,
        args.suffix,
        args.use_wandb,
        args.class_names,
    )
