import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from tqdm import tqdm
import torch.nn.functional as F
import os
import numpy as np
from model import CADTransformer
from dataset import create_dataloaders
import datetime
from loss import CADLoss, ValidationLoss
import json
import pandas as pd


class CADTransformerTrainer:
    def __init__(self, config, resume_from=None):
        """
        Args:
            config: Configuration dictionary
            resume_from: Path to checkpoint directory containing latest_model.pt and config.json
        """
        # If resuming, load config from checkpoint
        if resume_from:
            config_path = os.path.join(resume_from, 'config.json')
            checkpoint_path = os.path.join(resume_from, 'latest_model.pt')
            
            if not os.path.exists(config_path) or not os.path.exists(checkpoint_path):
                raise ValueError(f"Could not find config.json or latest_model.pt in {resume_from}")
            
            # Load saved config
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            
            # Update current config with saved config
            config.update(saved_config)
            
            print(f"Loaded config from {config_path}")
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.START_TOKEN = 1025
        self.END_TOKEN = 1028
        
        # Replace separate loss criteria with CADLoss
        self.criterion = CADLoss(
            pad_idx=config['pad_idx'],
            token_weight=config['token_loss_weight'],
            coord_weight=config['coord_loss_weight']
        )
        
        self.model = CADTransformer(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers']
        ).to(self.device)
        
        self.initial_lr = config['learning_rate']
        self.start_lr = 1e-5
        
        self.optimizer = Adam(
            self.model.parameters(), 
            lr=self.start_lr,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=1e-5
        )
        
        # Calculate steps based on epochs and dataloader length
        steps_per_epoch = config['steps_per_epoch']
        total_steps = config['epochs'] * steps_per_epoch
        self.warmup_steps = config['warmup_epochs'] * steps_per_epoch
        self.steady_steps = config['steady_epochs'] * steps_per_epoch

        cosine_steps = total_steps - self.warmup_steps - self.steady_steps
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cosine_steps,
            eta_min=1e-6
        )

        self.current_step = 0
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        
        # Load checkpoint if resuming
        if resume_from:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.current_step = self.start_epoch * steps_per_epoch
            self.best_val_loss = checkpoint['val_loss']
            
            print(f"Resumed from epoch {self.start_epoch}")
            print(f"Current step: {self.current_step}")
            print(f"Best validation loss: {self.best_val_loss}")
        
        self.save_dir = config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.val_criterion = ValidationLoss(pad_idx=config['pad_idx'])

    def adjust_learning_rate(self):
        if self.current_step < self.warmup_steps:
            progress = float(self.current_step) / float(self.warmup_steps)
            lr = self.start_lr + (self.initial_lr - self.start_lr) * progress
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        elif self.current_step < (self.warmup_steps + self.steady_steps):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.initial_lr
        else:
            self.scheduler.step()

        return self.optimizer.param_groups[0]['lr']

    def to_device(self, batch):
        return tuple(t.to(self.device) for t in batch)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_token_loss = 0
        total_coord_loss = 0
        
        for batch in tqdm(train_loader, desc='Training'):
            src, tgt, src_mask, tgt_mask = self.to_device(batch)
            
            current_lr = self.adjust_learning_rate()
            
            batch_size = src.size(0)
            seq_len = tgt.size(1)

            # Teacher forcing: use target sequence except last token as input
            tgt_input = tgt[:, :-1]  # Input sequence is [START, token1, token2, ...]
            
            self.optimizer.zero_grad()
            
            # Forward pass with full sequence
            logits = self.model(src, tgt_input, src_key_padding_mask=src_mask)
            
            # Target sequence is shifted by one position
            tgt_output = tgt[:, 1:]  # Output sequence is [token1, token2, ..., END]
            
            # Prepare inputs for loss
            logits_flat = logits.contiguous().view(-1, self.config['output_dim'])
            tgt_output_flat = tgt_output.contiguous().view(-1)
            
            # Calculate loss using CADLoss
            loss, token_loss, coord_loss = self.criterion(logits_flat, tgt_output_flat, epoch=self.current_step // len(train_loader))
            
            # Update totals
            total_loss += loss.item()
            total_token_loss += token_loss
            total_coord_loss += coord_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient check
            total_norm = torch.norm(torch.stack([
                p.grad.norm(2) for p in self.model.parameters() if p.grad is not None
            ]))

            print(f"Step {self.current_step}: Total Loss = {loss.item():.4f}, "
                  f"Token Loss = {token_loss:.4f}, Coord Loss = {coord_loss:.4f}, "
                  f"Total Gradient Norm = {total_norm:.4f}")
            
            has_explosion, has_vanishing = self.check_gradients()
            if has_explosion:
                print("WARNING: Gradient explosion detected!")
            if has_vanishing:
                print("WARNING: Gradient vanishing detected!")

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_grad'])
            self.optimizer.step()
            
            self.current_step += 1
            
        return total_loss / len(train_loader), total_token_loss, total_coord_loss

    def check_gradients(self):
        has_explosion = False
        has_vanishing = False
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(2).item()
                if grad_norm > 10.0:
                    has_explosion = True
                elif grad_norm < 1e-6:
                    has_vanishing = True
        return has_explosion, has_vanishing

    def validate(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0
        last_samples = []  # Store last two samples
        
        # Load existing validation samples if file exists
        samples_file = os.path.join(self.save_dir, 'validation_samples.npy')
        if os.path.exists(samples_file):
            all_samples = list(np.load(samples_file, allow_pickle=True))
        else:
            all_samples = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                src, tgt, src_mask, tgt_mask = self.to_device(batch)

                batch_size = src.size(0)
                max_length = tgt.size(1)

                # need to modify this part to do autoregressive decoding 
                tgt_input = torch.full((batch_size, 1), self.START_TOKEN, device=self.device)
                outputs = []
                logits_collected = []  

                # Generate one token less than target length since we don't need to predict after END
                for i in range(max_length - 1):
                    logits = self.model(src, tgt_input, src_mask)

                    next_token_logits = logits[:, -1, :]
                    logits_collected.append(next_token_logits)
                    
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                    tgt_input = torch.cat([tgt_input, next_token], dim=-1)

                    outputs.append(next_token)

                    if torch.all(next_token == self.END_TOKEN):
                        break
                
                # Stack collected logits
                logits = torch.stack(logits_collected, dim=1)  # (batch_size, seq_len, output_dim)

                # Get target sequence (excluding START token)
                tgt_output = tgt[:, 1:logits.size(1)+1]  # Only take as many targets as we have predictions

                # Flatten for loss calculation
                logits = logits.contiguous().view(-1, self.config['output_dim'])
                tgt_output = tgt_output.contiguous().view(-1)  # Target sequence, shifted by one

                # Use simple cross entropy loss for validation
                loss = self.val_criterion(logits, tgt_output)
                total_loss += loss.item()

                # Store sample information
                sample = {
                    'epoch': epoch,
                    'generated': torch.cat(outputs, dim=1)[0].cpu().tolist(),
                    'target': tgt[0, 1:].cpu().tolist(),
                    'loss': loss.item()
                }
                last_samples.append(sample)
                if len(last_samples) > 2:
                    last_samples.pop(0)
                
                # Add to all samples
                all_samples.append(sample)

            # Print last two samples
            print("\nLast two validation samples:")
            print("============================")
            for i, sample in enumerate(last_samples, 1):
                print(f"\nSample {i}:")
                print(f"Generated: {sample['generated']}")
                print(f"Target   : {sample['target']}")
                print(f"Loss: {sample['loss']:.4f}")
                print("----------------------------")
            
            # Save all samples to numpy file
            np.save(samples_file, np.array(all_samples, dtype=object))

            return total_loss / len(val_loader)

    def train(self, train_loader, val_loader):
        # Initialize history dictionary
        history_file = os.path.join(self.save_dir, 'training_history.csv')
        if os.path.exists(history_file) and self.start_epoch > 0:
            # Load existing history if resuming
            history_df = pd.read_csv(history_file)
            history = {
                'epoch': history_df['epoch'].tolist(),
                'train_loss': history_df['train_loss'].tolist(),
                'train_token_loss': history_df['train_token_loss'].tolist(),
                'train_coord_loss': history_df['train_coord_loss'].tolist(),
                'val_loss': history_df['val_loss'].tolist(),
                'learning_rate': history_df['learning_rate'].tolist()
            }
        else:
            history = {
                'epoch': [],
                'train_loss': [],
                'train_token_loss': [],
                'train_coord_loss': [],
                'val_loss': [],
                'learning_rate': []
            }
        
        # Start training from the appropriate epoch
        for epoch in range(self.start_epoch, self.config['epochs']):
            print(f"\nStarting epoch {epoch}/{self.config['epochs']}")
            # Create new dataloaders for each epoch with different augmentations
            # Pass the actual epoch number, not starting from 0
            train_loader, val_loader = create_dataloaders(self.config, epoch=epoch)
            
            # Training
            epoch_token_loss = 0
            epoch_coord_loss = 0
            train_loss, epoch_token_loss, epoch_coord_loss = self.train_epoch(train_loader)
            
            # Get average token and coord losses for the epoch
            epoch_token_loss /= len(train_loader)
            epoch_coord_loss /= len(train_loader)
            
            # Validation
            val_loss = self.validate(val_loader, epoch + 1)
            print(f"Epoch {epoch+1}/{self.config['epochs']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1}/{self.config['epochs']} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"LR: {current_lr:.6f}")
            
            # Update history
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(train_loss)
            history['train_token_loss'].append(epoch_token_loss)
            history['train_coord_loss'].append(epoch_coord_loss)
            history['val_loss'].append(val_loss)
            history['learning_rate'].append(current_lr)
            
            # Save checkpoint if validation loss improves
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'val_loss': val_loss,
                'config': self.config,
                'history': history
            }
            
            if val_loss < getattr(self, 'best_val_loss', float('inf')):
                self.best_val_loss = val_loss
                # Save best model
                torch.save(checkpoint, os.path.join(self.save_dir, 'best_model.pt'))
                
            # Save latest model
            torch.save(checkpoint, os.path.join(self.save_dir, 'latest_model.pt'))
            
            # Save training history to CSV
            df = pd.DataFrame(history)
            df.to_csv(os.path.join(self.save_dir, 'training_history.csv'), index=False)
            
            # Save config separately
            with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
                json.dump(self.config, f, indent=4)



if __name__ == "__main__":
    config = {
        'd_model': 512,
        'num_heads': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'learning_rate': 1e-4,
        'pad_idx': 1024,
        'input_dim': 16,
        'output_dim': 1036,
        'd_ff': 2048,
        'dropout': 0.1,
        'epochs': 20,
        'clip_grad': 5,
        'max_input_length': 512,
        'max_output_length': 512,
        'batch_size': 32,
        'val_batch_size': 1,
        'warmup_epochs': 3,
        'steady_epochs': 5,
        'token_loss_weight': 1.0,
        'coord_loss_weight': 0.1,
        'save_dir': 'checkpoints/011425',
    }
    
    resume_from = None
    
    # If resuming, get the starting epoch from checkpoint
    if resume_from:
        checkpoint_path = os.path.join(resume_from, 'latest_model.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            start_epoch = checkpoint['epoch'] + 1
            # Create initial dataloaders with correct epoch
            train_loader, val_loader = create_dataloaders(config, epoch=start_epoch)
    else:
        # For new training, start with epoch 0
        train_loader, val_loader = create_dataloaders(config, epoch=0)
    
    config['steps_per_epoch'] = len(train_loader)
    trainer = CADTransformerTrainer(config, resume_from=resume_from)
    trainer.train(train_loader, val_loader)
