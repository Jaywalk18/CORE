# solo/losses/core.py
import torch
import torch.nn.functional as F
from solo.utils.misc import gather


def core_contrastive_loss(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Computes CORE's contrastive loss given batch of projected features z1 and z2.
    Implements the essential contrastive loss using normalized features.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2 (often from momentum encoder).
        temperature (float, optional): temperature parameter for softmax. Defaults to 0.1.

    Returns:
        torch.Tensor: CORE contrastive loss.
    """
    # Ensure input tensors are 2D
    if z1.dim() > 2:
        z1 = z1.view(z1.size(0), -1)
    if z2.dim() > 2:
        z2 = z2.view(z2.size(0), -1)
    
    # Normalize feature vectors
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    
    # Gather features from all devices if using distributed training
    gathered_z2 = gather(z2)
    
    # Compute similarity scores
    sim = torch.exp(torch.einsum("if, jf -> ij", z1, gathered_z2) / temperature)
    
    # Create identity matrix for positive pairs
    batch_size = z1.size(0)
    pos_mask = torch.eye(batch_size, device=z1.device, dtype=torch.bool)
    
    # Expand mask if using distributed training
    world_size = gathered_z2.size(0) // batch_size
    if world_size > 1:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        pos_mask = torch.zeros(batch_size, gathered_z2.size(0), device=z1.device, dtype=torch.bool)
        pos_mask[:, rank * batch_size:(rank + 1) * batch_size] = torch.eye(batch_size, device=z1.device, dtype=torch.bool)
    
    # Compute positive and negative similarities
    pos = torch.sum(sim * pos_mask, dim=1)
    neg = torch.sum(sim * (~pos_mask), dim=1)
    
    # Compute the loss
    loss = -torch.mean(torch.log(pos / (pos + neg + 1e-8)))
    
    return loss


def core_loss_func(
    z1: torch.Tensor, 
    z2: torch.Tensor,
    momentum_z1: torch.Tensor = None,
    momentum_z2: torch.Tensor = None,
    temperature: float = 0.1
) -> dict:
    """CORE loss function implementing COntrastive Representation with Essential components.
    
    Uses bidirectional contrastive learning between views and their momentum counterparts
    when available, otherwise falls back to direct contrastive learning.
    
    Args:
        z1: Projected features from first view
        z2: Projected features from second view
        momentum_z1: Momentum encoder features from first view (optional)
        momentum_z2: Momentum encoder features from second view (optional)
        temperature: Temperature scaling parameter
        
    Returns:
        dict: Dictionary containing the loss components and total loss
    """
    loss = 0.0
    
    # Bidirectional momentum contrastive loss when momentum encoders are available
    if momentum_z2 is not None:
        loss += core_contrastive_loss(z1, momentum_z2, temperature)
    if momentum_z1 is not None:
        loss += core_contrastive_loss(z2, momentum_z1, temperature)
    
    # Fallback to direct contrastive learning if no momentum features
    if momentum_z1 is None and momentum_z2 is None:
        loss = core_contrastive_loss(z1, z2, temperature)
    
    return {"contrastive_loss": loss, "total_loss": loss}
