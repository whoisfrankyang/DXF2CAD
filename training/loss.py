import torch
import torch.nn as nn
import torch.nn.functional as F

class CADLoss(nn.Module):
    def __init__(self, pad_idx=1024, token_weight=1.0, coord_weight=0.1):
        super().__init__()
        self.pad_idx = pad_idx
        self.token_weight = token_weight
        self.coord_weight = coord_weight
        
        # Token loss (cross entropy for classification)
        self.token_criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        
        # Coordinate loss (Huber for regression)
        self.value_criterion = nn.HuberLoss(reduction='none', delta=1.0)

    def forward(self, logits_flat, tgt_output_flat, epoch=None):
        """
        Args:
            logits_flat: (batch_size * seq_len, output_dim) Raw model outputs
            tgt_output_flat: (batch_size * seq_len) Target tokens
            epoch: Current training epoch (for coordinate weight scheduling)
        """
        # Calculate token loss (cross entropy on full logits)
        token_loss = self.token_criterion(logits_flat, tgt_output_flat)
        
        # Coordinate regression for all positions
        coord_logits = logits_flat[:, :1024]  # First 1024 logits for coordinate regression
        coord_probs = torch.softmax(coord_logits, dim=-1)  # Probabilities over coordinates
        coord_values = torch.arange(1024, device=logits_flat.device, dtype=torch.float)
        
        # Compute expected coordinates for all positions
        predicted_coords = (coord_probs * coord_values).sum(dim=-1)
        
        # Prepare target coordinates
        coord_positions = (tgt_output_flat < 1024)
        target_coords = tgt_output_flat.clone().float()
        target_coords[~coord_positions] = 0  # Set non-coordinate targets to 0
        
        # Calculate Huber loss
        coord_loss = self.value_criterion(predicted_coords, target_coords)
        # Only normalize by actual coordinate positions
        coord_loss = coord_loss.sum() / (coord_positions.sum() + 1e-6)
        
        # Update coordinate weight based on epoch if provided
        current_coord_weight = min(0.5, epoch/50) if epoch is not None else self.coord_weight

        # Calculate weighted loss
        loss = self.token_weight * token_loss + current_coord_weight * coord_loss
        
        return loss, token_loss.item(), coord_loss.item() 

class ValidationLoss(nn.Module):
    def __init__(self, pad_idx):
        super().__init__()
        self.pad_idx = pad_idx
        self.criterion = nn.L1Loss(reduction='none')
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size * seq_len, output_dim) Raw model outputs
            targets: (batch_size * seq_len) Target tokens
        """
        # Get coordinate probabilities for values < 1024
        coord_logits = logits[:, :1024]  # First 1024 logits for coordinate regression
        coord_probs = torch.softmax(coord_logits, dim=-1)
        coord_values = torch.arange(1024, device=logits.device, dtype=torch.float)
        
        # Compute expected coordinates
        predicted_coords = (coord_probs * coord_values).sum(dim=-1)
        
        # Prepare target coordinates
        coord_positions = (targets < 1024)
        target_coords = targets.clone().float()
        target_coords[~coord_positions] = 0  # Set non-coordinate targets to 0
        
        # Calculate L1 loss only for coordinate positions
        loss = self.criterion(predicted_coords, target_coords)
        # Mask out non-coordinate positions and padding
        mask = coord_positions & (targets != self.pad_idx)
        loss = loss * mask.float()
        
        # Average over actual coordinate positions
        return loss.sum() / (mask.sum() + 1e-6) 