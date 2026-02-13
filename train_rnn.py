import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
import math
import sys
import os
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
import json  
import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(7)

def parse_coords(s):
    nums = re.findall(r"-?\d+", s)
    return tuple(map(int, nums)) if len(nums) == 2 else None

def extract_between(tag, text):
    patterns = [
        rf"<\s*{tag}\s*[\-\s]?\s*START\s*>(.*?)<\s*{tag}\s*[\-\s]?\s*END\s*>",
        rf"<\s*{tag}START\s*>(.*?)<\s*{tag}END\s*>",
        rf"<\s*{tag}\s*START\s*>(.*?)<\s*{tag}\s*END\s*>",
        rf"<\s*{tag.replace(' ', '')}\s*START\s*>(.*?)<\s*{tag.replace(' ', '')}\s*END\s*>",
    ]
    for p in patterns:
        m = re.search(p, text, re.S | re.I)
        if m:
            return m.group(1).strip()
    return "" 

def plot_maze(tokens, title="Maze"):
    text = " ".join(tokens)
    adj_section = extract_between("ADJLIST", text)
    origin_section = extract_between("ORIGIN", text)
    target_section = extract_between("TARGET", text)
    path_section = extract_between("PATH", text)
    origin = parse_coords(origin_section)
    target = parse_coords(target_section)
    edge_matches = re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)\s*<-->\s*\(\s*-?\d+\s*,\s*-?\d+\s*\)", adj_section)
    edges = []
    for em in edge_matches:
        coords = re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", em)
        a = parse_coords(coords[0])
        b = parse_coords(coords[1])
        edges.append((a, b))
    path = [parse_coords(p) for p in re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", path_section)]
    if not path and path_section:
        nums = re.findall(r"-?\d+\s*,\s*-?\d+", path_section)
        path = [tuple(map(int, re.findall(r"-?\d+", s))) for s in nums]
    if not edges:
        print("Warning: No edges found in adjacency list.")
        return
    if not origin or not target:
        print("Warning: Could not parse origin or target.")
        return
    rows = 6
    cols = 6
    vertical_walls = np.ones((rows, cols + 1), dtype=bool)
    horizontal_walls = np.ones((rows + 1, cols), dtype=bool)
    for (r1, c1), (r2, c2) in edges:
        if r1 == r2:
            c_between = min(c1, c2) + 1
            if 0 <= r1 < rows and 0 <= c_between <= cols:
                vertical_walls[r1, c_between] = False
        elif c1 == c2:
            r_between = min(r1, r2) + 1
            if 0 <= r_between <= rows and 0 <= c1 < cols:
                horizontal_walls[r_between, c1] = False
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect('equal')
    for r in range(rows + 1):
        ax.plot([0, cols], [r, r], color='lightgray', lw=1, zorder=0)
    for c in range(cols + 1):
        ax.plot([c, c], [0, rows], color='lightgray', lw=1, zorder=0)
    for r in range(rows):
        for c in range(cols + 1):
            if vertical_walls[r, c]:
                y_world = rows - r
                ax.plot([c, c], [y_world - 1, y_world], color='black', lw=4, solid_capstyle='butt')
    for r in range(rows + 1):
        for c in range(cols):
            if horizontal_walls[r, c]:
                y_world = rows - r
                ax.plot([c, c + 1], [y_world, y_world], color='black', lw=4, solid_capstyle='butt')
    if path:
        path_x = [c + 0.5 for (r, c) in path]
        path_y = [rows - r - 0.5 for (r, c) in path]
        ax.plot(path_x, path_y, linestyle='--', linewidth=2, color='red', zorder=4)
    ox, oy = origin[1] + 0.5, rows - origin[0] - 0.5
    tx, ty = target[1] + 0.5, rows - target[0] - 0.5
    ax.scatter(ox, oy, c='red', s=80, marker='o', zorder=5, label='Start')
    ax.scatter(tx, ty, c='blue', s=80, marker='x', zorder=5, label='End')
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(np.arange(cols + 1))
    ax.set_yticks(np.arange(rows + 1))
    ax.set_xticklabels([str(c) for c in range(cols + 1)])
    ax.set_yticklabels([str(rows - r) for r in range(rows + 1)])
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    plt.title(title)
    plt.tight_layout()
    plt.show()

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

def build_vocabulary(csv_path):
    df = pd.read_csv(csv_path)
    vocab = set()
    vocab.add(PAD_TOKEN)
    vocab.add(SOS_TOKEN)
    vocab.add(EOS_TOKEN)
    for index, row in df.iterrows():
        try:
            input_tokens = eval(row['input_sequence'])
            output_tokens = eval(row['output_path'])
            vocab.update(input_tokens)
            vocab.update(output_tokens)
        except Exception as e:
            print(f"Error reading row {index}: {e}")
            continue
    sorted_vocab = sorted(list(vocab))
    token_to_idx = {token: idx for idx, token in enumerate(sorted_vocab)}
    idx_to_token = {idx: token for idx, token in enumerate(sorted_vocab)}
    PAD_IDX = token_to_idx[PAD_TOKEN]
    return token_to_idx, idx_to_token, len(vocab), PAD_IDX

class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        inputs = [item[0] for item in batch]
        outputs = [item[1] for item in batch]
        padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=self.pad_idx)
        padded_outputs = pad_sequence(outputs, batch_first=True, padding_value=self.pad_idx)
        return padded_inputs, padded_outputs

class MazeDataset(Dataset):
    def __init__(self, csv_file, token_to_idx, sos_token, eos_token):
        self.df = pd.read_csv(csv_file)
        self.token_to_idx = token_to_idx
        self.sos_token = sos_token
        self.eos_token = eos_token

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        input_tokens = eval(row['input_sequence'])
        input_indices = [self.token_to_idx[token] for token in input_tokens]
        output_tokens = eval(row['output_path'])
        output_tokens_with_sos_eos = [self.sos_token] + output_tokens + [self.eos_token]
        output_indices = [self.token_to_idx[token] for token in output_tokens_with_sos_eos]
        input_tensor = torch.tensor(input_indices, dtype=torch.long).flip(dims=[0])
        output_tensor = torch.tensor(output_indices, dtype=torch.long)
        return input_tensor, output_tensor

import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_dim, n_layers, dropout, pad_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, src_tensor):
        embedded = self.dropout(self.embedding(src_tensor))
        encoder_outputs, hidden = self.rnn(embedded)
        return encoder_outputs, hidden

class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.attn_enc = nn.Linear(encoder_hidden_dim, decoder_hidden_dim)
        self.attn_dec = nn.Linear(decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)
    def forward(self, decoder_hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        src_seq_len = encoder_outputs.shape[1]
        dec_hidden_top = decoder_hidden[-1] 
        dec_hidden_repeated = dec_hidden_top.unsqueeze(1).repeat(1, src_seq_len, 1)
        energy = torch.tanh(
            self.attn_enc(encoder_outputs) + self.attn_dec(dec_hidden_repeated)
        )
        attention_scores = self.v(energy).squeeze(2)
        weights = F.softmax(attention_scores, dim=1)
        weights_expanded = weights.unsqueeze(1)
        context = torch.bmm(weights_expanded, encoder_outputs)
        context = context.squeeze(1)
        return context, weights

class DecoderRNN(nn.Module):
    def __init__(self, output_vocab_size, embedding_dim, 
                 encoder_hidden_dim, decoder_hidden_dim, n_layers, dropout, attention, pad_idx=None):
        super().__init__()
        self.output_vocab_size = output_vocab_size
        self.attention = attention
        self.n_layers = n_layers
        self.decoder_hidden_dim = decoder_hidden_dim
        self.embedding = nn.Embedding(output_vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(  
            input_size=embedding_dim + encoder_hidden_dim,
            hidden_size=decoder_hidden_dim,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(decoder_hidden_dim, output_vocab_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, dec_input_token, dec_hidden, encoder_outputs):
        dec_input_token = dec_input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(dec_input_token))
        context, attn_weights = self.attention(dec_hidden, encoder_outputs)
        context = context.unsqueeze(1)
        rnn_input = torch.cat((embedded, context), dim=2)
        dec_output, dec_hidden = self.rnn(rnn_input, dec_hidden)
        prediction = dec_output.squeeze(1)
        prediction = self.fc_out(prediction)
        return prediction, dec_hidden, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, pad_idx=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    def forward(self, src_tensor, trg_tensor, teacher_forcing_ratio=0.5):
        batch_size = trg_tensor.shape[0]
        trg_len = trg_tensor.shape[1]
        trg_vocab_size = self.decoder.output_vocab_size
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src_tensor)
        dec_input = trg_tensor[:, 0]
        for t in range(1, trg_len):
            dec_output, hidden, _ = self.decoder(dec_input, hidden, encoder_outputs)
            outputs[:, t] = dec_output
            use_teacher_force = random.random() < teacher_forcing_ratio
            top1 = dec_output.argmax(1)
            dec_input = trg_tensor[:, t] if use_teacher_force else top1
        return outputs

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
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
        self.embedding = nn.Embedding(input_vocab_size, d_model) 
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True 
        )
        self.fc_out = nn.Linear(d_model, output_vocab_size)
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)
    def _create_padding_mask(self, tensor):
        return (tensor == self.pad_idx).to(self.device)
    def _embed_and_position(self, tensor):
        embedded = self.embedding(tensor) * math.sqrt(self.d_model)
        embedded_pos = self.pos_encoder(embedded.permute(1, 0, 2))
        return embedded_pos.permute(1, 0, 2)
    def forward(self, src_tensor, trg_tensor):
        src_padding_mask = self._create_padding_mask(src_tensor)
        trg_padding_mask = self._create_padding_mask(trg_tensor)
        trg_len = trg_tensor.shape[1]
        trg_subsequent_mask = self._generate_square_subsequent_mask(trg_len)
        src_embedded = self._embed_and_position(src_tensor)
        trg_embedded = self._embed_and_position(trg_tensor)
        output = self.transformer(
            src=src_embedded,
            tgt=trg_embedded,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=trg_padding_mask,
            tgt_mask=trg_subsequent_mask
        )
        prediction = self.fc_out(output)
        return prediction


CHOOSE_MODEL = 'rnn'  

TRAIN_CSV_PATH = '/kaggle/input/assignment-4-task-1-2/train_6x6_mazes.csv'
TEST_CSV_PATH  = '/kaggle/input/assignment-4-task-1-2/test_6x6_mazes.csv'

if not os.path.exists(TRAIN_CSV_PATH):
    print(f"Error: Training file not found at {TRAIN_CSV_PATH}")
    sys.exit()
if not os.path.exists(TEST_CSV_PATH):
    print(f"Error: Test file not found at {TEST_CSV_PATH}")
    sys.exit()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

BATCH_SIZE = 32
N_EPOCHS = 20
LEARNING_RATE = 1e-4
CLIP = 1.0 

RNN_EMBEDDING_DIM = 128
RNN_HIDDEN_DIM = 512
RNN_N_LAYERS = 2
RNN_DROPOUT = 0.1

TR_D_MODEL = 128
TR_NHEAD = 8
TR_NUM_ENCODER_LAYERS = 6
TR_NUM_DECODER_LAYERS = 6
TR_DIM_FEEDFORWARD = 512
TR_DROPOUT = 0.1

print("Building vocabulary...")
token_to_idx, idx_to_token, VOCAB_SIZE, PAD_IDX = build_vocabulary(TRAIN_CSV_PATH)
INPUT_VOCAB_SIZE = VOCAB_SIZE
OUTPUT_VOCAB_SIZE = VOCAB_SIZE
SOS_IDX = token_to_idx[SOS_TOKEN]
EOS_IDX = token_to_idx[EOS_TOKEN]

print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"PAD token index: {PAD_IDX}")

print("Saving vocabulary...")
with open('vocab.json', 'w') as f:
    json.dump(token_to_idx, f)

print("Setting up DataLoaders...")
collate_fn = Collate(pad_idx=PAD_IDX)
full_dataset = MazeDataset(
    csv_file=TRAIN_CSV_PATH,
    token_to_idx=token_to_idx,
    sos_token=SOS_TOKEN,
    eos_token=EOS_TOKEN
)

total_size = len(full_dataset)
valid_size = int(total_size * 0.1) 
train_size = total_size - valid_size
print(f"Full dataset size: {total_size}")
print(f"Splitting into: {train_size} training examples, {valid_size} validation examples")
train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)
valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

print("Setting up Test Loader...")
test_dataset = MazeDataset(
    csv_file=TEST_CSV_PATH,
    token_to_idx=token_to_idx,
    sos_token=SOS_TOKEN,
    eos_token=EOS_TOKEN
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

print(f"Initializing Model: {CHOOSE_MODEL.upper()}")

if CHOOSE_MODEL == 'rnn':
    attention_module = Attention(
        encoder_hidden_dim=RNN_HIDDEN_DIM,
        decoder_hidden_dim=RNN_HIDDEN_DIM
    )
    encoder_rnn = EncoderRNN(
        input_vocab_size=INPUT_VOCAB_SIZE,
        embedding_dim=RNN_EMBEDDING_DIM,
        hidden_dim=RNN_HIDDEN_DIM,
        n_layers=RNN_N_LAYERS,
        dropout=RNN_DROPOUT,
        pad_idx=PAD_IDX
    )
    decoder_rnn = DecoderRNN(
        output_vocab_size=OUTPUT_VOCAB_SIZE,
        embedding_dim=RNN_EMBEDDING_DIM,
        encoder_hidden_dim=RNN_HIDDEN_DIM,
        decoder_hidden_dim=RNN_HIDDEN_DIM,
        n_layers=RNN_N_LAYERS,
        dropout=RNN_DROPOUT,
        attention=attention_module,
        pad_idx=PAD_IDX
    )
    model = Seq2Seq(encoder_rnn, decoder_rnn, DEVICE).to(DEVICE)

elif CHOOSE_MODEL == 'transformer':
    model = TransformerSeq2Seq(
        input_vocab_size=INPUT_VOCAB_SIZE,
        output_vocab_size=OUTPUT_VOCAB_SIZE,
        d_model=TR_D_MODEL,
        nhead=TR_NHEAD,
        num_encoder_layers=TR_NUM_ENCODER_LAYERS,
        num_decoder_layers=TR_NUM_DECODER_LAYERS,
        dim_feedforward=TR_DIM_FEEDFORWARD,
        dropout=TR_DROPOUT,
        pad_idx=PAD_IDX,
        device=DEVICE
    ).to(DEVICE)
else:
    print(f"Error: Unknown model type '{CHOOSE_MODEL}'. Choose 'rnn' or 'transformer'.")
    sys.exit()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX) 

def calculate_metrics(predictions, targets, pad_idx):
    top_predictions = predictions.argmax(2)
    non_pad_mask = (targets != pad_idx)
    correct_tokens = (top_predictions == targets) & non_pad_mask
    num_correct_tokens = correct_tokens.sum()
    num_non_pad_tokens = non_pad_mask.sum()
    token_accuracy = num_correct_tokens.float() / num_non_pad_tokens if num_non_pad_tokens > 0 else 0.0
    incorrect_tokens_in_seq = (top_predictions != targets) & non_pad_mask
    seq_has_error = incorrect_tokens_in_seq.sum(dim=1) > 0
    correct_sequences = (seq_has_error == False).sum()
    total_sequences = targets.shape[0]
    sequence_accuracy = correct_sequences.float() / total_sequences
    f1_score = token_accuracy  
    return token_accuracy.item(), sequence_accuracy.item(), f1_score.item()

def train_epoch(model, loader, optimizer, criterion, clip, model_type, pad_idx):
    model.train()
    epoch_loss = 0
    epoch_token_acc = 0
    epoch_seq_acc = 0
    epoch_f1 = 0
    
    for i, (src_tensor, trg_tensor) in enumerate(loader):
        src_tensor = src_tensor.to(DEVICE)
        trg_tensor = trg_tensor.to(DEVICE)
        
        optimizer.zero_grad()
        
        if model_type == 'rnn':
            output = model(src_tensor, trg_tensor, teacher_forcing_ratio=0.5)
            output_for_loss = output[:, 1:].reshape(-1, output.shape[-1])
            trg_for_loss = trg_tensor[:, 1:].reshape(-1)
            output_for_acc = output[:, 1:]
            trg_for_acc = trg_tensor[:, 1:]
            
        elif model_type == 'transformer':
            trg_input = trg_tensor[:, :-1]
            trg_target = trg_tensor[:, 1:]
            output = model(src_tensor, trg_input)
            output_for_loss = output.reshape(-1, output.shape[-1])
            trg_for_loss = trg_target.reshape(-1)
            output_for_acc = output
            trg_for_acc = trg_target

        loss = criterion(output_for_loss, trg_for_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
        token_acc, seq_acc, f1 = calculate_metrics(output_for_acc, trg_for_acc, pad_idx)
        epoch_token_acc += token_acc
        epoch_seq_acc += seq_acc
        epoch_f1 += f1
        
    return epoch_loss / len(loader), epoch_token_acc / len(loader), epoch_seq_acc / len(loader), epoch_f1 / len(loader)

def evaluate_epoch(model, loader, criterion, model_type, pad_idx):
    model.eval()
    epoch_loss = 0
    epoch_token_acc = 0
    epoch_seq_acc = 0
    epoch_f1 = 0
    
    with torch.no_grad():
        for i, (src_tensor, trg_tensor) in enumerate(loader):
            src_tensor = src_tensor.to(DEVICE)
            trg_tensor = trg_tensor.to(DEVICE)
            
            if model_type == 'rnn':
                output = model(src_tensor, trg_tensor, 0.0)
                output_for_loss = output[:, 1:].reshape(-1, output.shape[-1])
                trg_for_loss = trg_tensor[:, 1:].reshape(-1)
                output_for_acc = output[:, 1:]
                trg_for_acc = trg_tensor[:, 1:]
                
            elif model_type == 'transformer':
                trg_input = trg_tensor[:, :-1]
                trg_target = trg_tensor[:, 1:]
                output = model(src_tensor, trg_input)
                output_for_loss = output.reshape(-1, output.shape[-1])
                trg_for_loss = trg_target.reshape(-1)
                output_for_acc = output
                trg_for_acc = trg_target

            loss = criterion(output_for_loss, trg_for_loss)
            epoch_loss += loss.item()
            
            token_acc, seq_acc, f1 = calculate_metrics(output_for_acc, trg_for_acc, pad_idx)
            epoch_token_acc += token_acc
            epoch_seq_acc += seq_acc
            epoch_f1 += f1
            
    return epoch_loss / len(loader), epoch_token_acc / len(loader), epoch_seq_acc / len(loader), epoch_f1 / len(loader)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

best_valid_seq_acc = 0.0
SAVED_MODEL_PATH = f'{CHOOSE_MODEL}-bonus-model.pt'

print(f"\n--- Starting Training for {CHOOSE_MODEL.upper()} (BONUS RUN) ---")

train_losses, valid_losses = [], []
train_token_accs, valid_token_accs = [], []
train_seq_accs, valid_seq_accs = [], []
train_f1s, valid_f1s = [], [] 

test_losses, test_token_accs, test_seq_accs, test_f1s = [], [], [], []  

for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    train_loss, train_token_acc, train_seq_acc, train_f1 = train_epoch(
        model, train_loader, optimizer, criterion, CLIP, CHOOSE_MODEL, PAD_IDX
    )
    valid_loss, valid_token_acc, valid_seq_acc, valid_f1 = evaluate_epoch(
        model, valid_loader, criterion, CHOOSE_MODEL, PAD_IDX
    )
    test_loss_epoch, test_token_acc_epoch, test_seq_acc_epoch, test_f1_epoch = evaluate_epoch(  
        model, test_loader, criterion, CHOOSE_MODEL, PAD_IDX                                
    )                                                                                       
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    # store train/val
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_token_accs.append(train_token_acc)
    valid_token_accs.append(valid_token_acc)
    train_seq_accs.append(train_seq_acc)
    valid_seq_accs.append(valid_seq_acc)
    train_f1s.append(train_f1) 
    valid_f1s.append(valid_f1) 

    # store test
    test_losses.append(test_loss_epoch)                
    test_token_accs.append(test_token_acc_epoch)       
    test_seq_accs.append(test_seq_acc_epoch)           
    test_f1s.append(test_f1_epoch)                     
    
    if valid_seq_acc > best_valid_seq_acc:
        best_valid_seq_acc = valid_seq_acc
        torch.save(model.state_dict(), SAVED_MODEL_PATH)
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
    print(f'\tTrain Token Acc: {train_token_acc*100:.2f}% | Train Seq Acc: {train_seq_acc*100:.2f}% | Train F1: {train_f1:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):.3f}')
    print(f'\t Val. Token Acc: {valid_token_acc*100:.2f}% | Val. Seq Acc: {valid_seq_acc*100:.2f}% | Val. F1: {valid_f1:.3f}')
    print(f'\t Test Loss: {test_loss_epoch:.3f} | Test Token Acc: {test_token_acc_epoch*100:.2f}% | Test Seq Acc: {test_seq_acc_epoch*100:.2f}% | Test F1: {test_f1_epoch:.3f}')  # NEW

print("--- Training Finished ---")
def predict_path(model, src_tensor, max_len=30, model_type='rnn'):
    model.eval()
    
    with torch.no_grad():
        if model_type == 'rnn':
            encoder_outputs, hidden = model.encoder(src_tensor)
            trg_indices = [SOS_IDX]
            
            for _ in range(max_len):
                trg_tensor = torch.LongTensor([trg_indices[-1]]).to(DEVICE)
                
                output, hidden, _ = model.decoder(trg_tensor, hidden, encoder_outputs)
                pred_token = output.argmax(1).item()
                trg_indices.append(pred_token)
                
                if pred_token == EOS_IDX:
                    break
        
        elif model_type == 'transformer':
            src_mask = model._create_padding_mask(src_tensor)
            src_embedded = model.embedding(src_tensor) * math.sqrt(model.d_model) 
            src_embedded = model.pos_encoder(src_embedded.permute(1, 0, 2)).permute(1, 0, 2)
            
            encoder_output = model.transformer.encoder(
                src_embedded, 
                src_key_padding_mask=src_mask
            )

            trg_indices = [SOS_IDX]
            
            for _ in range(max_len):
                trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(DEVICE)
                
                trg_len = trg_tensor.shape[1]
                trg_mask = model._generate_square_subsequent_mask(trg_len)
                
                trg_embedded = model.embedding(trg_tensor) * math.sqrt(model.d_model) 
                trg_embedded = model.pos_encoder(trg_embedded.permute(1, 0, 2)).permute(1, 0, 2)
                
                decoder_output = model.transformer.decoder(
                    trg_embedded, 
                    encoder_output, 
                    tgt_mask=trg_mask,
                    memory_key_padding_mask=src_mask 
                )
                
                pred_token = model.fc_out(decoder_output[:, -1]).argmax(1).item()
                trg_indices.append(pred_token)

                if pred_token == EOS_IDX:
                    break

    return [idx_to_token[idx] for idx in trg_indices]

print("\n" + "="*50)
print(f"STARTING FINAL EVALUATION ON TEST SET ({CHOOSE_MODEL.upper()})")
print("="*50 + "\n")

try:
    model.load_state_dict(torch.load(SAVED_MODEL_PATH))
    print(f"Successfully loaded best model from {SAVED_MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Could not find saved model at {SAVED_MODEL_PATH}")
    print(f"Make sure the file {SAVED_MODEL_PATH} was saved in /kaggle/working/")
    sys.exit()

test_loss, test_token_acc, test_seq_acc, test_f1 = evaluate_epoch(
    model, test_loader, criterion, CHOOSE_MODEL, PAD_IDX
)

print("\n" + "---" * 10)
print("FINAL TEST SET RESULTS (Best Val Seq-Acc Model):")
print(f"\tTest Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}")
print(f"\tTest Token Acc: {test_token_acc*100:.2f}% | Test Seq Acc: {test_seq_acc*100:.2f}% | Test F1: {test_f1:.3f}")
print("---" * 10 + "\n")

print("--- Generating Plots ---")
epochs_range = range(1, N_EPOCHS + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, valid_losses, label='Validation Loss')
plt.plot(epochs_range, test_losses,  label='Test Loss')           
plt.title(f'{CHOOSE_MODEL.upper()} - Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(epochs_range, [acc * 100 for acc in train_token_accs], label='Train Token Accuracy')
plt.plot(epochs_range, [acc * 100 for acc in valid_token_accs], label='Validation Token Accuracy')
plt.plot(epochs_range, [acc * 100 for acc in test_token_accs],  label='Test Token Accuracy')    
plt.title(f'{CHOOSE_MODEL.upper()} - Token Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(epochs_range, [acc * 100 for acc in train_seq_accs], label='Train Sequence Accuracy')
plt.plot(epochs_range, [acc * 100 for acc in valid_seq_accs], label='Validation Sequence Accuracy')
plt.plot(epochs_range, [acc * 100 for acc in test_seq_accs],  label='Test Sequence Accuracy')    
plt.title(f'{CHOOSE_MODEL.upper()} - Sequence Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(epochs_range, train_f1s, label='Train F1-Score')
plt.plot(epochs_range, valid_f1s, label='Validation F1-Score')
plt.plot(epochs_range, test_f1s,  label='Test F1-Score')        
plt.title(f'{CHOOSE_MODEL.upper()} - F1-Score vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('F1-Score')
plt.legend()
plt.show()

print("--- Test Set Inference Examples (with Visualization) ---")
num_examples_to_show = 5
for i in range(num_examples_to_show):
    if i >= len(test_dataset):
        break 
    
    src_test, trg_test = test_dataset[i]
    src_tensor = src_test.unsqueeze(0).to(DEVICE)
    
    src_tokens = [idx_to_token[idx.item()] for idx in src_test]
    ground_truth_path = [idx_to_token[idx.item()] for idx in trg_test]
    predicted_path = predict_path(model, src_tensor, 30, CHOOSE_MODEL)
    
    print(f"\n--- Example {i+1} ---")
    print(f"Ground Truth (Text): {' '.join(ground_truth_path)}")
    print(f"Prediction (Text):   {' '.join(predicted_path)}")
    
    truth_plot_tokens = src_tokens + ["<PATH_START>"] + ground_truth_path[1:-1] + ["<PATH_END>"]
    plot_maze(truth_plot_tokens, title=f"Example {i+1}: Ground Truth")

    pred_path_tokens = predicted_path[1:-1] if predicted_path[-1] == EOS_TOKEN else predicted_path[1:]
    pred_plot_tokens = src_tokens + ["<PATH_START>"] + pred_path_tokens + ["<PATH_END>"]
    plot_maze(pred_plot_tokens, title=f"Example {i+1}: Model Prediction")

print("\n--- Evaluation Complete ---")
