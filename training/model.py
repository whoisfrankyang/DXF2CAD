import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (TransformerDecoder, TransformerDecoderLayer,
                     TransformerEncoder, TransformerEncoderLayer)
import math
from utils import quantize_values


"""
Some notes about mask:
1. src_key_padding_mask: coming from dataloader, created in collate_fn in dataset.py and are passed in as a parameter
 in forward function in train.py
2. tgt_key_padding_mask: this is automatically generated in forward function in model.py by setting tgt == pad_idx
3. causal_mask: this is automatically generated in forward function in model.py by using _generate_square_subsequent_mask
4. loss_mask: this is created in train.py 
"""
class CADTransformer(nn.Module):
    def __init__(self, 
                input_dim=13,
                output_dim=1033,
                d_model=512, 
                num_heads=8, 
                num_encoder_layers=6, 
                num_decoder_layers=6,
                d_ff=2048,
                dropout=0.1,
                max_input_length=512,
                max_output_length=512,
                pad_idx=1024):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.output_dim = output_dim

        # Input embeddings and projections
        self.input_embeddings = nn.ModuleDict({
            'input_value': nn.Embedding(1025, d_model),  # 1024 values + 1 padding
            'input_view': nn.Linear(6, d_model, bias=False),  # Remove bias since it's one-hot
            'input_vis': nn.Linear(2, d_model, bias=False),   # Remove bias since it's one-hot
            'input_coord_type': nn.Linear(4, d_model, bias=False),  # Remove bias since it's one-hot
            "input_dimension": nn.Linear(3, d_model, bias=False),  # Remove bias since it's one-hot
        })
        
        # Target embedding
        self.target_embedding = nn.Embedding(output_dim, d_model)
        
        # Separate positional encodings for input and output
        self.input_pos_encoding = PositionalEncoding(d_model, max_input_length)
        self.output_pos_encoding = PositionalEncoding(d_model, max_output_length)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        # Transformer decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        
        # Two projection heads (tokens and values)
        self.output_head = nn.Linear(d_model, output_dim)

        # Initialize parameters
        self._reset_parameters()
        
    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.0)
            elif 'bias' in name:
                nn.init.zeros_(p)  # Zero for biases
                
    def _generate_square_subsequent_mask(self, sz):
        """Generate causal mask for transformer"""
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
        
        
    def forward(self, src, tgt, src_key_padding_mask=None, causal_mask=None, tgt_key_padding_mask=None):

        # Embed value, view, vis, coord_type
        value = src[:, :, 0].long()
        value_embed = self.input_embeddings['input_value'](value)
        
        # Direct linear projections for one-hot vectors
        view = src[:, :, 1:7]
        vis = src[:, :, 7:9]
        coord_type = src[:, :, 9:13]
        dimension = src[:, :, 13:16]
        
        # Linear projections
        view_embed = self.input_embeddings['input_view'](view)
        vis_embed = self.input_embeddings['input_vis'](vis)
        coord_type_embed = self.input_embeddings['input_coord_type'](coord_type)
        dimension_embed = self.input_embeddings['input_dimension'](dimension)

        # Combine embeddings
        src_embedded = (
            value_embed +  # Embedded quantized coordinate
            view_embed +
            vis_embed +
            coord_type_embed +
            dimension_embed
        )
        
        # Add positional encoding
        src_embedded = self.input_pos_encoding(src_embedded)
        src_embedded = self.dropout(src_embedded)
        
        # Target embeddings 
        tgt_embedded = self.target_embedding(tgt)
        tgt_embedded = self.output_pos_encoding(tgt_embedded)
        tgt_embedded = self.dropout(tgt_embedded)
        


        if src_key_padding_mask is not None:
            src_key_padding_mask = ~src_key_padding_mask.bool()
        else:
            src_key_padding_mask = None

        tgt_key_padding_mask = (tgt == self.pad_idx)


        sz = tgt.size(1)
        if causal_mask is None:
            causal_mask = self._generate_square_subsequent_mask(sz).to(tgt.device)


        # Encoder
        memory = self.encoder(
            src_embedded, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Decoder
        output = self.decoder(
            tgt_embedded, 
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        logits = self.output_head(output)
        
        # The target/output is padded with 1024 
        # The padded output positions is set to a very small value to ensure 
        # they do not affect the loss calculation 
        logits[:, :, self.pad_idx] = -1e2
        

        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=5000):
        super().__init__()
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

def build_model(config):
    return CADTransformer(
        input_dim=config.get('input_dim', 16),
        output_dim=config.get('output_dim', 1036),
        d_model=config.get('d_model', 512),
        num_heads=config.get('num_heads', 8),
        num_encoder_layers=config.get('num_encoder_layers', 6),
        num_decoder_layers=config.get('num_decoder_layers', 6),
        d_ff=config.get('d_ff', 2048),
        dropout=config.get('dropout', 0.1),
        max_input_length=config.get('max_input_length', 512),
        max_output_length=config.get('max_output_length', 512),
        pad_idx=config.get('pad_idx', 1024)
    )