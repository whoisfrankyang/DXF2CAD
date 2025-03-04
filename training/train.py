import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from tqdm import tqdm
import wandb

from model import Transformer
from dataset import create_dataloaders, CADDataset

class CADTransformerTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = Transformer(
            input_dim=19,
            output_dim=1,
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers']
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=config['pad_idx'])
        self.optimizer = Adam(
            self.model.parameters(), 
            lr=config['learning_rate'],
            betas=(0.9, 0.98),
            eps=1e-9
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=config['scheduler_patience'],
            factor=0.5
        )
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc='Training'):
            src, tgt, src_mask, tgt_mask = batch
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            src_mask = src_mask.to(self.device)
            tgt_mask = tgt_mask.to(self.device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            self.optimizer.zero_grad()
            output = self.model(src, tgt_input, src_mask, tgt_mask)
            
            loss = self.criterion(output, tgt_output)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_grad'])
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                src, tgt, src_mask, tgt_mask = batch
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                src_mask = src_mask.to(self.device)
                tgt_mask = tgt_mask.to(self.device)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                output = self.model(src, tgt_input, src_mask, tgt_mask)
                loss = self.criterion(output, tgt_output)
                
                total_loss += loss.item()
                
        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, save_dir='checkpoints'):
        best_val_loss = float('inf')
        early_stopping_counter = 0
        
        wandb.init(project=self.config['project_name'], config=self.config)
        
        for epoch in range(self.config['epochs']):
            start_time = time.time()
            
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                self.save_checkpoint(f'{save_dir}/best_model.pt')
            else:
                early_stopping_counter += 1
            
            if early_stopping_counter >= self.config['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            epoch_time = time.time() - start_time
            print(f"Epoch: {epoch+1}/{self.config['epochs']}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Epoch time: {epoch_time:.2f}s")
            
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            })

    def save_checkpoint(self, filename):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, filename)
    
    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['config']

if __name__ == "__main__":
    config = {
        'd_model': 512,
        'num_heads': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'learning_rate': 1e-4,
        'batch_size': 32,
        'epochs': 100,
        'clip_grad': 1.0,
        'scheduler_patience': 3,
        'early_stopping_patience': 10,
        'project_name': 'cad_transformer'
    }
    
    train_dataset = CADDataset(
        data_path='path/to/train/data',
        max_length=512,
        input_vec_dim=19,
        target_vec_dim=1
    )
    
    val_dataset = CADDataset(
        data_path='path/to/val/data',
        max_length=512,
        input_vec_dim=19,
        target_vec_dim=1
    )
    
    train_loader, val_loader = create_dataloaders(
        train_dataset, 
        val_dataset,
        batch_size=config['batch_size']
    )
    
    trainer = CADTransformerTrainer(config)
    trainer.train(train_loader, val_loader)