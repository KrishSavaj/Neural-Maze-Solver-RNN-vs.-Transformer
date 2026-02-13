import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import math
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
import time

TRAIN_CSV_PATH = '/kaggle/input/assignment-4-task-1-2/train_6x6_mazes.csv'
TEST_CSV_PATH = '/kaggle/input/assignment-4-task-1-2/test_6x6_mazes.csv'
MODEL_SAVE_PATH = 'maze_transformer_strict.pt'
SUBMISSION_PATH = 'submission.csv'

D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

BATCH_SIZE = 128        
N_EPOCHS = 35           
LEARNING_RATE = 0.0005
SEQ_LEN = 100           

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {DEVICE}")

PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'

class MazeVocabulary:
    def __init__(self, tokens):
        self.idx_to_token = {i: t for i, t in enumerate(tokens)}
        self.token_to_idx = {t: i for i, t in enumerate(tokens)}
        self.pad_idx = self.token_to_idx.get(PAD_TOKEN, 0)
        self.unk_idx = self.token_to_idx.get(UNK_TOKEN, 0)

    def __len__(self): 
        return len(self.idx_to_token)
    
    def encode(self, t_list): 
        return [self.token_to_idx.get(t, self.unk_idx) for t in t_list]
    
    def decode(self, i_list): 
        return [self.idx_to_token.get(i, UNK_TOKEN) for i in i_list]


def build_vocab(csv_path):
    print("Building Vocabulary...")
    df = pd.read_csv(csv_path)
    tokens = set([PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN])
    for _, row in df.iterrows():
        tokens.update(ast.literal_eval(row['input_sequence']))
        if 'output_path' in row: 
            tokens.update(ast.literal_eval(row['output_path']))
    return MazeVocabulary(sorted(list(tokens)))


class MazeDataset(Dataset):
    def __init__(self, csv_path, vocab, is_train=True, max_len=100):
        print(f"Pre-loading {('Train/Labelled' if is_train else 'Unlabelled Test')} Data...")
        df = pd.read_csv(csv_path)
        self.vocab = vocab
        self.max_len = max_len
        self.is_train = is_train
        self.data = []

        for _, row in df.iterrows():
            src = ast.literal_eval(row['input_sequence'])
            src_ids = self.vocab.encode(src)
            
            if self.is_train:
                tgt = ast.literal_eval(row['output_path'])
                tgt_ids = self.vocab.encode([SOS_TOKEN] + tgt + [EOS_TOKEN])
                self.data.append((src_ids, tgt_ids))
            else:
                self.data.append((src_ids, []))
        print("Data loaded.")

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        
        src_t = torch.tensor(src, dtype=torch.long)
        if len(src_t) < self.max_len:
            src_t = torch.cat(
                [src_t, torch.full((self.max_len - len(src_t),), self.vocab.pad_idx, dtype=torch.long)]
            )
        else: 
            src_t = src_t[:self.max_len]
            
        if self.is_train:
            tgt_t = torch.tensor(tgt, dtype=torch.long)
            if len(tgt_t) < self.max_len:
                tgt_t = torch.cat(
                    [tgt_t, torch.full((self.max_len - len(tgt_t),), self.vocab.pad_idx, dtype=torch.long)]
                )
            else: 
                tgt_t = tgt_t[:self.max_len]
            return src_t, tgt_t
        
        return src_t, torch.tensor([], dtype=torch.long)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, D_MODEL)
        # self.pe: (max_len, 1, D_MODEL)
        x = x + self.pe[:x.size(1), :].transpose(0, 1)  # match (B, T, D)
        return self.dropout(x)


class MazeTransformer(nn.Module):
    def __init__(self, vocab_size, pad_idx):
        super().__init__()
        
        self.pad_idx = pad_idx
        
        self.embedding = nn.Embedding(vocab_size, D_MODEL)
        
        self.pos_encoder = SinusoidalPositionalEncoding(D_MODEL, DROPOUT)
        
        self.transformer = nn.Transformer(
            d_model=D_MODEL,
            nhead=NHEAD,
            num_encoder_layers=NUM_LAYERS,
            num_decoder_layers=NUM_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(D_MODEL, vocab_size)
        self.fc_out.weight = self.embedding.weight
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return mask.to(DEVICE)

    def forward(self, src, tgt):
        src_mask = (src == self.pad_idx).to(DEVICE)
        tgt_mask = (tgt == self.pad_idx).to(DEVICE)
        tgt_causal_mask = self._generate_square_subsequent_mask(tgt.size(1))
        
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(D_MODEL))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(D_MODEL))
        
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_mask,
            tgt_mask=tgt_causal_mask
        )
        return self.fc_out(output)  # (B, T, V)


def calculate_metrics(preds, targets, pad_idx):
    batch_size = preds.size(0)
    mask = (targets != pad_idx)
    
    correct = ((preds == targets) & mask).sum().item()
    total = mask.sum().item()
    tok_acc = correct / total if total > 0 else 0.0
    
    seq_match = ((preds == targets) | (~mask)).all(dim=1)
    seq_acc = seq_match.sum().item() / batch_size
    
    f1_sum = 0.0
    for i in range(batch_size):
        p_seq = set(preds[i][mask[i]].cpu().numpy())
        t_seq = set(targets[i][mask[i]].cpu().numpy())
        if len(p_seq) + len(t_seq) == 0:
            f1_sum += 1.0
        elif len(p_seq) == 0 or len(t_seq) == 0:
            f1_sum += 0.0
        else:
            common = len(p_seq & t_seq)
            f1_sum += (2 * common) / (len(p_seq) + len(t_seq))
            
    return tok_acc, seq_acc, f1_sum / batch_size

def run_training():
    if not os.path.exists(TRAIN_CSV_PATH):
        print("Training CSV not found.")
        return None

    # --- Vocabulary ---
    vocab = build_vocab(TRAIN_CSV_PATH)

    # --- Datasets ---
    full_ds = MazeDataset(TRAIN_CSV_PATH, vocab, is_train=True, max_len=SEQ_LEN)
    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    
    test_ds = MazeDataset(TEST_CSV_PATH, vocab, is_train=True, max_len=SEQ_LEN)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Vocab: {len(vocab)} | Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    model = MazeTransformer(len(vocab), vocab.pad_idx).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.001, 
        steps_per_epoch=len(train_loader), 
        epochs=N_EPOCHS
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    print("\nStarting Training...")
    best_seq = 0.0

    train_losses, val_losses, test_losses = [], [], []
    train_tok_accs, val_tok_accs, test_tok_accs = [], [], []
    train_seq_accs, val_seq_accs, test_seq_accs = [], [], []
    
    for epoch in range(1, N_EPOCHS + 1):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        epoch_train_tok, epoch_train_seq = [], []
        
        for src, tgt in train_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            
            optimizer.zero_grad()
            output = model(src, tgt_in)
            loss = criterion(output.reshape(-1, len(vocab)), tgt_out.reshape(-1))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            preds = output.argmax(dim=-1)
            t_a, s_a, _ = calculate_metrics(preds, tgt_out, vocab.pad_idx)
            epoch_train_tok.append(t_a)
            epoch_train_seq.append(s_a)
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_tok  = sum(epoch_train_tok) / len(epoch_train_tok)
        avg_train_seq  = sum(epoch_train_seq) / len(epoch_train_seq)
        
        model.eval()
        val_loss = 0.0
        val_tok_list, val_seq_list = [], []
        
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
                
                output = model(src, tgt_in)
                loss = criterion(output.reshape(-1, len(vocab)), tgt_out.reshape(-1))
                val_loss += loss.item()
                
                preds = output.argmax(dim=-1)
                t_a, s_a, _ = calculate_metrics(preds, tgt_out, vocab.pad_idx)
                val_tok_list.append(t_a)
                val_seq_list.append(s_a)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_tok  = sum(val_tok_list) / len(val_tok_list)
        avg_val_seq  = sum(val_seq_list) / len(val_seq_list)

        test_loss = 0.0
        test_tok_list, test_seq_list = [], []
        
        with torch.no_grad():
            for src, tgt in test_loader:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
                
                output = model(src, tgt_in)
                loss = criterion(output.reshape(-1, len(vocab)), tgt_out.reshape(-1))
                test_loss += loss.item()
                
                preds = output.argmax(dim=-1)
                t_a, s_a, _ = calculate_metrics(preds, tgt_out, vocab.pad_idx)
                test_tok_list.append(t_a)
                test_seq_list.append(s_a)

        avg_test_loss = test_loss / len(test_loader)
        avg_test_tok  = sum(test_tok_list) / len(test_tok_list)
        avg_test_seq  = sum(test_seq_list) / len(test_seq_list)
        
        curr_lr = optimizer.param_groups[0]['lr']
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        test_losses.append(avg_test_loss)

        train_tok_accs.append(avg_train_tok)
        val_tok_accs.append(avg_val_tok)
        test_tok_accs.append(avg_test_tok)

        train_seq_accs.append(avg_train_seq)
        val_seq_accs.append(avg_val_seq)
        test_seq_accs.append(avg_test_seq)

        if avg_val_seq > best_seq:
            best_seq = avg_val_seq
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        
        end_time = time.time()
        epoch_mins = int((end_time - start_time) / 60)
        epoch_secs = int((end_time - start_time) % 60)

        print(f"Epoch {epoch:02} | Time: {epoch_mins}m {epoch_secs}s | LR: {curr_lr:.6f}")
        print(f"  Train: Loss={avg_train_loss:.3f}, TokAcc={avg_train_tok*100:.2f}%, SeqAcc={avg_train_seq*100:.2f}%")
        print(f"  Val:   Loss={avg_val_loss:.3f}, TokAcc={avg_val_tok*100:.2f}%, SeqAcc={avg_val_seq*100:.2f}%")
        print(f"  Test:  Loss={avg_test_loss:.3f}, TokAcc={avg_test_tok*100:.2f}%, SeqAcc={avg_test_seq*100:.2f}%")
        print("-" * 70)
    
    epochs = range(1, N_EPOCHS + 1)

    # Loss curves
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses,   label='Val Loss')
    plt.plot(epochs, test_losses,  label='Test Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs (Train / Val / Test)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(epochs, [x*100 for x in train_tok_accs], label='Train Token Acc')
    plt.plot(epochs, [x*100 for x in val_tok_accs],   label='Val Token Acc')
    plt.plot(epochs, [x*100 for x in test_tok_accs],  label='Test Token Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Token Accuracy (%)")
    plt.title("Token Accuracy vs Epochs (Train / Val / Test)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(epochs, [x*100 for x in train_seq_accs], label='Train Seq Acc')
    plt.plot(epochs, [x*100 for x in val_seq_accs],   label='Val Seq Acc')
    plt.plot(epochs, [x*100 for x in test_seq_accs],  label='Test Seq Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Sequence Accuracy (%)")
    plt.title("Sequence Accuracy vs Epochs (Train / Val / Test)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return vocab

def run_inference(vocab):
    print("\nStarting Inference on UNLABELLED test (for submission)...")
    if not os.path.exists(TEST_CSV_PATH): 
        print("Test CSV not found.")
        return

    model = MazeTransformer(len(vocab), vocab.pad_idx).to(DEVICE)
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    else:
        print("Warning: model weights not found, using random model.")
    
    model.eval()
    test_ds = MazeDataset(TEST_CSV_PATH, vocab, is_train=False, max_len=SEQ_LEN)
    sos_idx = vocab.token_to_idx[SOS_TOKEN]
    eos_idx = vocab.token_to_idx[EOS_TOKEN]
    preds = []
    
    print(f"Predicting {len(test_ds)} samples...")
    with torch.no_grad():
        for i in range(len(test_ds)):
            src, _ = test_ds[i]
            src = src.unsqueeze(0).to(DEVICE)
            
            src_emb = model.pos_encoder(model.embedding(src) * math.sqrt(D_MODEL))
            memory = model.transformer.encoder(src_emb)
            
            tgt_idxs = [sos_idx]
            for _ in range(50):
                tgt = torch.tensor([tgt_idxs], dtype=torch.long).to(DEVICE)
                tgt_emb = model.pos_encoder(model.embedding(tgt) * math.sqrt(D_MODEL))
                tgt_mask = model._generate_square_subsequent_mask(len(tgt_idxs))
                out = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                next_tok = model.fc_out(out[:, -1]).argmax(-1).item()
                tgt_idxs.append(next_tok)
                if next_tok == eos_idx: 
                    break
            
            tokens = vocab.decode(tgt_idxs)
            clean = [t for t in tokens if t not in [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN]]
            preds.append(str(clean))
            
            if i % 1000 == 0: 
                print(f"Predicted {i}...")

    df = pd.read_csv(TEST_CSV_PATH)
    df['predicted_path'] = preds
    df.to_csv(SUBMISSION_PATH, index=False)
    print("Submission saved to:", SUBMISSION_PATH)
    
    if len(preds) > 0:
        simple_plot_maze(ast.literal_eval(preds[0]), "Test Example 1 (Predicted Path)")


def simple_plot_maze(token_list, title):
    coords = []
    for t in token_list:
        try:
            if t.startswith('('): 
                coords.append(ast.literal_eval(t))
        except: 
            pass

    if not coords: 
        return

    arr = np.array(coords)
    plt.figure(figsize=(4,4))
    plt.plot(arr[:,0], arr[:,1], '-o')
    if len(arr) > 0:
        plt.plot(arr[0,0], arr[0,1], 'sg')
        plt.plot(arr[-1,0], arr[-1,1], 'Xr')
    plt.title(title)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    vocab = run_training()
    if vocab:
        run_inference(vocab)
