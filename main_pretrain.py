# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
Main script for self-supervised pretraining using Lightning.
Supports multiple logging backends (SwanLab, Weights & Biases, TensorBoard) for seamless experiment tracking and visualization.
"""
import inspect
import os
from typing import Dict, Any, Union, Optional

import time
import torch
from lightning.pytorch.callbacks import Callback

import hydra
import torch
import swanlab
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers import TensorBoardLogger, Logger as LightningLoggerBase
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.strategies.ddp import DDPStrategy

from solo.args.pretrain import parse_cfg
from solo.data.classification_dataloader import prepare_data as prepare_data_classification
from solo.data.pretrain_dataloader import (
    FullTransformPipeline,
    NCropAugmentation,
    build_transform_pipeline,
    prepare_dataloader,
    prepare_datasets,
)
from solo.methods import METHODS
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import Checkpointer
from solo.utils.misc import make_contiguous, omegaconf_select

# Optional imports for DALI data loading
try:
    from solo.data.dali_dataloader import PretrainDALIDataModule, build_transform_pipeline_dali
    _dali_avaliable = True
except ImportError:
    _dali_avaliable = False

# Optional imports for UMAP visualization
try:
    from solo.utils.auto_umap import AutoUMAP
    _umap_available = True
except ImportError:
    _umap_available = False


class SwanLabLogger(LightningLoggerBase):
    """
    SwanLab Logger integration for PyTorch Lightning.
    Enables experiment tracking with SwanLab.
    """
    def __init__(
        self,
        project: str = "default",
        workspace: str = "CORE-SSL",
        experiment_name: Optional[str] = None,
        # resume = False,
        save_dir: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        self._project = project
        self._workspace = workspace
        self._name = experiment_name
        self._save_dir = save_dir
        self._kwargs = kwargs
        # self._resume=resume
        self._experiment = None
        self._run = self._init_experiment()
        
    def _init_experiment(self):
        """Initialize SwanLab experiment."""
        return swanlab.init(
            project=self._project,
            workspace=self._workspace,
            experiment_name=self._name,
            save_dir=self._save_dir,
            # resume=self._resume,
            # id=self._id,
            **self._kwargs
        )
    
    @property
    def experiment(self):
        """Return the SwanLab experiment instance."""
        if self._experiment is None:
            self._experiment = self._run
        return self._experiment
    
    @rank_zero_only
    def log_hyperparams(self, params: Dict[str, Any]):
        """Log hyperparameters to SwanLab."""
        self.experiment.config.update(params)
    
    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Union[float, torch.Tensor]], step: Optional[int] = None):
        """Log metrics to SwanLab with optional step information."""
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            
            if step is not None:
                self.experiment.log({k: v}, step=step)
            else:
                self.experiment.log({k: v})
    
    @property
    def name(self) -> str:
        """Return the experiment name."""
        return self._name if self._name else "default"
    
    @property
    def version(self) -> Union[str, int]:
        """Return the experiment version/run ID."""
        return self.experiment.run_id if hasattr(self.experiment, "run_id") else "0"
    
    def save(self):
        """Save method (currently a no-op as SwanLab handles saving automatically)."""
        pass


class ResourceMonitorCallback(Callback):
    """
    Callback to log resource usage per epoch: time, max CUDA memory, and throughput.
    Compatible with PyTorch Lightning loggers such as SwanLab.
    """
    
    def on_train_epoch_start(self, trainer, pl_module):
        # Record the start time at the beginning of each epoch
        pl_module._epoch_start_time = time.time()
        # Reset max memory stat for the upcoming epoch
        torch.cuda.reset_max_memory_allocated()

    def on_train_epoch_end(self, trainer, pl_module):
        # Calculate time spent in this epoch
        epoch_time = time.time() - pl_module._epoch_start_time
        # Calculate the peak CUDA memory usage (in MB) during this epoch
        max_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

        current_epoch = trainer.current_epoch

        # Approximate throughput (images per second for the current epoch)
        try:
            num_samples = len(trainer.train_dataloader.dataset)
            throughput = num_samples / epoch_time if epoch_time > 0 else 0
        except Exception:
            throughput = 0  # fallback if DataLoader doesn't provide dataset

        # Prepare metrics dictionary
        metrics = {
            "epoch_time_sec": epoch_time,
            "max_cuda_mem_MB": max_mem,
            "throughput_imgs_per_sec": throughput,
        }

        # Log metrics to SwanLab or any compatible logger
        if hasattr(pl_module, "logger") and pl_module.logger is not None:
            pl_module.logger.log_metrics(metrics, step=current_epoch)


@hydra.main(version_base="1.2")

def main(cfg: DictConfig):
    """
    Main function for self-supervised pretraining.
    
    Args:
        cfg: Configuration from Hydra
    """
    # Allow adding new configuration parameters
    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)
    
    # Set seed for reproducibility
    seed_everything(cfg.seed)
    
    # Validate method selection
    assert cfg.method in METHODS, f"Method '{cfg.method}' not found. Choose from: {list(METHODS.keys())}"
    
    # Validate crop configuration
    if cfg.data.num_large_crops != 2:
        assert cfg.method in ["wmse", "mae", "objcon"], \
            "Only wmse, mae and objcon support num_large_crops != 2"
    
    # Initialize model
    model = METHODS[cfg.method](cfg)
    make_contiguous(model)
    
    # Apply memory optimization if enabled
    if not cfg.performance.disable_channel_last:
        model = model.to(memory_format=torch.channels_last)  # ~20% speed up
    
    # Set up validation dataloader if available
    val_loader = None
    if not ((cfg.data.dataset == "custom" and (cfg.data.no_labels or cfg.data.val_path is None)) or
            (cfg.data.dataset in ["imagenet100", "imagenet"] and cfg.data.val_path is None)):
        
        val_data_format = "image_folder" if cfg.data.format == "dali" else cfg.data.format
        _, val_loader = prepare_data_classification(
            cfg.data.dataset,
            train_data_path=cfg.data.train_path,
            val_data_path=cfg.data.val_path,
            data_format=val_data_format,
            batch_size=cfg.optimizer.batch_size,
            num_workers=cfg.data.num_workers,
        )
    
    # Configure data loading
    if cfg.data.format == "dali":
        # Use DALI data loading pipeline
        assert _dali_avaliable, "DALI not available. Install with: pip install .[dali]"
        
        # Build augmentation pipelines
        pipelines = []
        for aug_cfg in cfg.augmentations:
            pipelines.append(
                NCropAugmentation(
                    build_transform_pipeline_dali(
                        cfg.data.dataset, aug_cfg, dali_device=cfg.dali.device
                    ),
                    aug_cfg.num_crops,
                )
            )
        transform = FullTransformPipeline(pipelines)
        
        # Create DALI datamodule
        dali_datamodule = PretrainDALIDataModule(
            dataset=cfg.data.dataset,
            train_data_path=cfg.data.train_path,
            transforms=transform,
            num_large_crops=cfg.data.num_large_crops,
            num_small_crops=cfg.data.num_small_crops,
            num_workers=cfg.data.num_workers,
            batch_size=cfg.optimizer.batch_size,
            no_labels=cfg.data.no_labels,
            data_fraction=cfg.data.fraction,
            dali_device=cfg.dali.device,
            encode_indexes_into_labels=cfg.dali.encode_indexes_into_labels,
        )
        dali_datamodule.val_dataloader = lambda: val_loader
    else:
        # Standard PyTorch data loading
        pipelines = []
        for aug_cfg in cfg.augmentations:
            pipelines.append(
                NCropAugmentation(
                    build_transform_pipeline(cfg.data.dataset, aug_cfg), 
                    aug_cfg.num_crops
                )
            )
        transform = FullTransformPipeline(pipelines)
        
        # Debug augmentations if requested
        if cfg.debug_augmentations:
            print("Transforms:")
            print(transform)
        
        # Create dataset and dataloader
        train_dataset = prepare_datasets(
            cfg.data.dataset,
            transform,
            train_data_path=cfg.data.train_path,
            data_format=cfg.data.format,
            no_labels=cfg.data.no_labels,
            data_fraction=cfg.data.fraction,
        )
        train_loader = prepare_dataloader(
            train_dataset, 
            batch_size=cfg.optimizer.batch_size, 
            num_workers=cfg.data.num_workers
        )
    
    # Handle checkpointing and resuming
    ckpt_path, wandb_run_id = None, None
    if cfg.auto_resume.enabled and cfg.resume_from_checkpoint is None:
        auto_resumer = AutoResumer(
            checkpoint_dir=os.path.join(cfg.checkpoint.dir, cfg.method),
            max_hours=cfg.auto_resume.max_hours,
        )
        resume_from_checkpoint, wandb_run_id = auto_resumer.find_checkpoint(cfg)
        if resume_from_checkpoint is not None:
            print(f"Resuming from previous checkpoint: '{resume_from_checkpoint}'")
            ckpt_path = resume_from_checkpoint
    elif cfg.resume_from_checkpoint is not None:
        ckpt_path = cfg.resume_from_checkpoint
        del cfg.resume_from_checkpoint
    
    # Set up callbacks
    callbacks = []
    
    # Checkpointing callback
    if cfg.checkpoint.enabled:
        ckpt = Checkpointer(
            cfg,
            logdir=os.path.join(cfg.checkpoint.dir, cfg.method),
            frequency=cfg.checkpoint.frequency,
            keep_prev=cfg.checkpoint.keep_prev,
        )
        callbacks.append(ckpt)
        callbacks.append(ResourceMonitorCallback())  
    # UMAP visualization callback
    if omegaconf_select(cfg, "auto_umap.enabled", False):
        assert _umap_available, "UMAP not available. Install with: pip install .[umap]"
        auto_umap = AutoUMAP(
            cfg.name,
            logdir=os.path.join(cfg.auto_umap.dir, cfg.method),
            frequency=cfg.auto_umap.frequency,
        )
        callbacks.append(auto_umap)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # Configure logger based on settings
    logger = None
    has_swanlab_config = hasattr(cfg, 'swanlab')
    
    if has_swanlab_config and cfg.swanlab.enabled:
        # Use SwanLab logger
        logger = SwanLabLogger(
            project=cfg.swanlab.project if hasattr(cfg.swanlab, 'project') else "default",
            experiment_name=cfg.name,
            save_dir=cfg.swanlab.log_dir if hasattr(cfg.swanlab, 'log_dir') else "./swanlab_logs"
        )
        logger.log_hyperparams(OmegaConf.to_container(cfg))
    elif cfg.wandb.enabled:
        # Use Weights & Biases logger
        logger = WandbLogger(
            name=cfg.name,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            offline=cfg.wandb.offline,
            resume="allow" if wandb_run_id else None,
            id=wandb_run_id,
        )
        logger.watch(model, log="gradients", log_freq=100)
        logger.log_hyperparams(OmegaConf.to_container(cfg))
    elif hasattr(cfg, 'tensorboard') and cfg.tensorboard.enabled:
        # Use TensorBoard logger
        logger = TensorBoardLogger(
            save_dir=cfg.tensorboard.log_dir,
            name=cfg.name,
            version=None,
            default_hp_metric=False
        )
        logger.log_hyperparams(OmegaConf.to_container(cfg))
    
    # Prepare Trainer configuration
    trainer_kwargs = OmegaConf.to_container(cfg)
    
    # Filter for valid Trainer arguments
    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {name: trainer_kwargs[name] for name in valid_kwargs if name in trainer_kwargs}
    
    # Configure distributed training strategy
    if cfg.strategy == "ddp":
        try:
            strategy = DDPStrategy(find_unused_parameters=False)
        except TypeError:  # Fallback for older PyTorch Lightning versions
            strategy = "ddp"
    else:
        strategy = cfg.strategy
    
    # Update trainer kwargs with logger and callbacks
    trainer_kwargs.update({
        "logger": logger,
        "callbacks": callbacks,
        "enable_checkpointing": False,
        "strategy": strategy,
    })
    
    # Initialize trainer
    trainer = Trainer(**trainer_kwargs)
    
    # Start training
    if cfg.data.format == "dali":
        trainer.fit(model, ckpt_path=ckpt_path, datamodule=dali_datamodule)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    # Entry point
    main()
