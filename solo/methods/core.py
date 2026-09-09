# solo/methods/core.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Sequence, Tuple
import omegaconf
from solo.methods.base import BaseMomentumMethod
from solo.losses.core import core_contrastive_loss, core_loss_func
from solo.utils.misc import omegaconf_select
from solo.utils.momentum import initialize_momentum_params
import os

class CoreProjectionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 2048,
        out_dim: int = 256,
    ):
        """Projection head for CORE (COntrastive Representation with Essential components).
      
        Args:
            in_dim (int): input dimension
            hidden_dim (int): hidden dimension
            out_dim (int): output dimension
        """
        super().__init__()
      
        self.projection = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class CORE(BaseMomentumMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements CORE (COntrastive Representation with Essential components).
        A minimal approach to contrastive learning that focuses only on the essential components.
        """
        super().__init__(cfg)

        # Visualization config remains unchanged
        self.visualization_enabled = omegaconf_select(cfg, "visualization.enabled", False)
        self.visualization_frequency = omegaconf_select(cfg, "visualization.frequency", 1)
        self.num_visualization_images = omegaconf_select(cfg, "visualization.num_images", 4)
        self.visualization_save_dir = omegaconf_select(cfg, "visualization.save_dir", 
                                                       os.path.join(cfg.checkpoint.dir, "visualization"))
        
        # Configuration parameters
        proj_hidden_dim = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim = cfg.method_kwargs.proj_output_dim
        self.temperature = cfg.method_kwargs.temperature
        
        # Only keep projection head
        self.projector = CoreProjectionHead(
            in_dim=self.features_dim,
            hidden_dim=proj_hidden_dim,
            out_dim=proj_output_dim
        )
        
        # Momentum projection head
        self.momentum_projector = CoreProjectionHead(
            in_dim=self.features_dim,
            hidden_dim=proj_hidden_dim,
            out_dim=proj_output_dim
        )
        
        # Initialize momentum parameters
        initialize_momentum_params(self.projector, self.momentum_projector)

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """
        cfg = super(CORE, CORE).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")

        # Set default values if not specified
        cfg.method_kwargs.temperature = omegaconf_select(cfg, "method_kwargs.temperature", 0.1)
        cfg.method_kwargs.num_classes = omegaconf_select(cfg, "method_kwargs.num_classes", 1000)

        return cfg


    @property
    def learnable_params(self) -> List[dict]:
        """Adds learnable parameters to the parent's learnable parameters."""
        extra_params = [
            {"name": "projector", "params": self.projector.parameters()},
        ]
        return super().learnable_params + extra_params


    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds momentum pairs."""
        extra_momentum_pairs = [
            (self.projector, self.momentum_projector),
        ]
        return super().momentum_pairs + extra_momentum_pairs


    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Forward pass through the backbone and projector."""
        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        
        # Use backbone
        feats = self.backbone(X)
        
        # Get features
        if isinstance(feats, dict):
            feats = feats.get("feats", feats)
        
        # Ensure feats is 2D
        if feats.dim() > 2:
            proj_feats = feats.view(feats.size(0), -1)
        else:
            proj_feats = feats
            
        proj = self.projector(proj_feats)
        
        # Generate logits if classifier exists
        if hasattr(self, 'classifier') and self.classifier is not None:
            logits = self.classifier(feats)
        else:
            logits = torch.zeros(feats.shape[0], getattr(self, 'num_classes', 1000), device=feats.device)
        
        return {
            "feats": feats,
            "z": proj,
            "logits": logits,
        }


    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Momentum forward method."""
        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        
        feats = self.momentum_backbone(X)
        
        if isinstance(feats, dict):
            feats = feats.get("feats", feats)
        
        if feats.dim() > 2:
            proj_feats = feats.view(feats.size(0), -1)
        else:
            proj_feats = feats
            
        proj = self.momentum_projector(proj_feats)
        
        if hasattr(self, 'momentum_classifier') and self.momentum_classifier is not None:
            logits = self.momentum_classifier(feats)
        else:
            logits = torch.zeros(feats.shape[0], getattr(self, 'num_classes', 1000), device=feats.device)
        
        return {
            "feats": feats,
            "z": proj,
            "logits": logits,
        }


    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for CORE."""
        
        out = super().training_step(batch, batch_idx)
        class_loss = out.get("loss", 0)
        
        # Get features and projections
        feats = out.get("feats", [])
        z = out.get("z", [])
        
        if not isinstance(feats, list):
            feats = [feats]
        if not isinstance(z, list):
            z = [z]
            
        # Get momentum projections
        momentum_out = out.get("momentum_out", {})
        momentum_z = momentum_out.get("z", [None, None])
        
        # Ensure we have at least 2 views
        if len(z) >= 2:
            z1, z2 = z[0], z[1]
            
            # Get momentum features
            momentum_z1 = momentum_z[0] if len(momentum_z) > 0 and momentum_z[0] is not None else None
            momentum_z2 = momentum_z[1] if len(momentum_z) > 1 and momentum_z[1] is not None else None
            
            # Calculate contrastive loss
            contrastive_loss = 0.0
            
            # If momentum features available, use bidirectional contrastive loss
            if momentum_z1 is not None and momentum_z2 is not None:
                contrastive_loss = core_contrastive_loss(z1, momentum_z2, self.temperature)
                contrastive_loss += core_contrastive_loss(z2, momentum_z1, self.temperature)
                contrastive_loss /= 2  # Average
            else:
                # Direct contrastive loss between views
                contrastive_loss = core_contrastive_loss(z1, z2, self.temperature)
            
            # Calculate total loss
            total_loss = contrastive_loss
            if class_loss != 0:
                total_loss = total_loss + class_loss

            # Log losses
            self.log("train_contrastive_loss", contrastive_loss, on_epoch=True, sync_dist=True)
            self.log("train_total_loss", total_loss, on_epoch=True, sync_dist=True)
            
            return total_loss
        
        # If not enough views, return class_loss
        return class_loss if class_loss != 0 else torch.tensor(0.0, device=self.device)
        
    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """Validation step for CORE.
        
        Args:
            batch: a batch of data in the format of [img_indexes, [X], Y]
            batch_idx: index of the batch
            
        Returns:
            Dict[str, Any]: dict with the batch_size and logits
        """
        # Just use parent class validation step
        return super().validation_step(batch, batch_idx)
        
    def configure_optimizers(self):
        """Configure optimizers for CORE.
        
        Returns:
            Tuple[List, List]: tuple of optimizer and scheduler
        """
        # Just use parent class optimizer configuration
        return super().configure_optimizers()
