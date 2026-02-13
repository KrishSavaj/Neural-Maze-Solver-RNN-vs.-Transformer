import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerSeq2Seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, d_model, nhead, 
                 num_encoder_layers, num_decoder_layers, dim_feedforward, 
                 dropout, pad_idx, device):
        
        super().__init__()
        
        self.pad_idx = pad_idx
        self.device = device
        self.d_model = d_model

        # 1. Embeddings (Source and Target)
        self.src_embedding = nn.Embedding(input_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(output_vocab_size, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 3. The core nn.Transformer module
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True 
        )
        
        # 4. Final output layer
        self.fc_out = nn.Linear(d_model, output_vocab_size)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)

    def _create_padding_mask(self, tensor):
        return (tensor == self.pad_idx).to(self.device)


    def forward(self, src_tensor, trg_tensor):
        
        # 1. Source Padding Mask
        src_padding_mask = self._create_padding_mask(src_tensor)
        
        # 2. Target Padding Mask
        trg_padding_mask = self._create_padding_mask(trg_tensor)
        
        # 3. Target Subsequent Mask (Look-ahead)
        trg_len = trg_tensor.shape[1]
        trg_subsequent_mask = self._generate_square_subsequent_mask(trg_len)
        
        
        src_embedded = self.src_embedding(src_tensor) * math.sqrt(self.d_model)
        src_embedded = self.pos_encoder(src_embedded.permute(1, 0, 2)).permute(1, 0, 2) # Handle batch_first
        
        trg_embedded = self.tgt_embedding(trg_tensor) * math.sqrt(self.d_model)
        trg_embedded = self.pos_encoder(trg_embedded.permute(1, 0, 2)).permute(1, 0, 2) # Handle batch_first
        
        output = self.transformer(
            src=src_embedded,
            tgt=trg_embedded,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=trg_padding_mask,
            tgt_mask=trg_subsequent_mask
        )
        prediction = self.fc_out(output)
        
        return prediction