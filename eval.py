import sys
import os
import ast
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# =====================================================
# 0. CONSTANTS & DEVICE
# =====================================================

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

# RNN hyperparams (must match training)
RNN_EMBEDDING_DIM = 128
RNN_HIDDEN_DIM = 512
RNN_N_LAYERS = 2
RNN_DROPOUT = 0.1

# Transformer hyperparams (must match training)
TR_D_MODEL = 128
TR_NHEAD = 8
TR_NUM_ENCODER_LAYERS = 6
TR_NUM_DECODER_LAYERS = 6
TR_DIM_FEEDFORWARD = 512
TR_DROPOUT = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# 1. HARDCODED VOCABS (PASTE YOUR JSON CONTENTS HERE)
# =====================================================

# Replace the {...} with the actual content of your RNN vocab.json
VOCAB_RNN = {"(0,0)": 0, "(0,1)": 1, "(0,2)": 2, "(0,3)": 3, "(0,4)": 4, "(0,5)": 5, "(1,0)": 6, "(1,1)": 7, "(1,2)": 8, "(1,3)": 9, "(1,4)": 10, "(1,5)": 11, "(2,0)": 12, "(2,1)": 13, "(2,2)": 14, "(2,3)": 15, "(2,4)": 16, "(2,5)": 17, "(3,0)": 18, "(3,1)": 19, "(3,2)": 20, "(3,3)": 21, "(3,4)": 22, "(3,5)": 23, "(4,0)": 24, "(4,1)": 25, "(4,2)": 26, "(4,3)": 27, "(4,4)": 28, "(4,5)": 29, "(5,0)": 30, "(5,1)": 31, "(5,2)": 32, "(5,3)": 33, "(5,4)": 34, "(5,5)": 35, ";": 36, "<-->": 37, "<ADJLIST_END>": 38, "<ADJLIST_START>": 39, "<EOS>": 40, "<ORIGIN_END>": 41, "<ORIGIN_START>": 42, "<PAD>": 43, "<PATH_END>": 44, "<PATH_START>": 45, "<SOS>": 46, "<TARGET_END>": 47, "<TARGET_START>": 48}

# Replace the {...} with the actual content of your Transformer vocab.json
VOCAB_TRANSFORMER = {"(0,0)": 0, "(0,1)": 1, "(0,2)": 2, "(0,3)": 3, "(0,4)": 4, "(0,5)": 5, "(1,0)": 6, "(1,1)": 7, "(1,2)": 8, "(1,3)": 9, "(1,4)": 10, "(1,5)": 11, "(2,0)": 12, "(2,1)": 13, "(2,2)": 14, "(2,3)": 15, "(2,4)": 16, "(2,5)": 17, "(3,0)": 18, "(3,1)": 19, "(3,2)": 20, "(3,3)": 21, "(3,4)": 22, "(3,5)": 23, "(4,0)": 24, "(4,1)": 25, "(4,2)": 26, "(4,3)": 27, "(4,4)": 28, "(4,5)": 29, "(5,0)": 30, "(5,1)": 31, "(5,2)": 32, "(5,3)": 33, "(5,4)": 34, "(5,5)": 35, ";": 36, "<-->": 37, "<ADJLIST_END>": 38, "<ADJLIST_START>": 39, "<EOS>": 40, "<ORIGIN_END>": 41, "<ORIGIN_START>": 42, "<PAD>": 43, "<PATH_END>": 44, "<PATH_START>": 45, "<SOS>": 46, "<TARGET_END>": 47, "<TARGET_START>": 48, "<UNK>": 49}


# =====================================================
# 2. MODEL DEFINITIONS (MATCH YOUR TRAINING CODE)
# =====================================================

# ---------- RNN SEQ2SEQ WITH ATTENTION ----------

class EncoderRNN(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_dim,
                 n_layers, dropout, pad_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim,
                                      padding_idx=pad_idx)
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
        # encoder_outputs: [batch, src_len, enc_hidden]
        # decoder_hidden:  [num_layers, batch, dec_hidden]
        batch_size = encoder_outputs.shape[0]
        src_seq_len = encoder_outputs.shape[1]

        dec_hidden_top = decoder_hidden[-1]  # [batch, dec_hidden]
        dec_hidden_repeated = dec_hidden_top.unsqueeze(1).repeat(1, src_seq_len, 1)

        energy = torch.tanh(
            self.attn_enc(encoder_outputs) + self.attn_dec(dec_hidden_repeated)
        )
        attention_scores = self.v(energy).squeeze(2)  # [batch, src_len]
        weights = F.softmax(attention_scores, dim=1)  # [batch, src_len]

        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [batch, enc_hidden]
        return context, weights


class DecoderRNN(nn.Module):
    def __init__(self, output_vocab_size, embedding_dim,
                 encoder_hidden_dim, decoder_hidden_dim,
                 n_layers, dropout, attention, pad_idx=None):
        super().__init__()
        self.output_vocab_size = output_vocab_size
        self.attention = attention
        self.embedding = nn.Embedding(output_vocab_size, embedding_dim,
                                      padding_idx=pad_idx)

        self.rnn = nn.RNN(
            input_size=embedding_dim + encoder_hidden_dim,
            hidden_size=decoder_hidden_dim,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True
        )

        self.fc_out = nn.Linear(decoder_hidden_dim, output_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input_token, hidden, encoder_outputs):
        # dec_input_token: [batch] -> [batch,1]
        dec_input_token = dec_input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(dec_input_token))  # [batch,1,emb]

        context, attn_weights = self.attention(hidden, encoder_outputs)
        context = context.unsqueeze(1)  # [batch,1,enc_hidden]

        rnn_input = torch.cat((embedded, context), dim=2)  # [batch,1,emb+enc_hidden]
        output, hidden = self.rnn(rnn_input, hidden)
        prediction = self.fc_out(output.squeeze(1))  # [batch,vocab]
        return prediction, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, pad_idx=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src_tensor, trg_tensor, teacher_forcing_ratio=0.5):
        # Not really needed in eval, but we keep it for completeness
        batch_size = trg_tensor.shape[0]
        trg_len = trg_tensor.shape[1]
        trg_vocab_size = self.decoder.output_vocab_size

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src_tensor)
        dec_input = trg_tensor[:, 0]

        for t in range(1, trg_len):
            dec_output, hidden, _ = self.decoder(dec_input, hidden, encoder_outputs)
            outputs[:, t] = dec_output
            top1 = dec_output.argmax(1)
            dec_input = top1

        return outputs


# ---------- TRANSFORMER (MazeTransformer STYLE) ----------

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
        x = x + self.pe[:x.size(1), :].transpose(0, 1)  # broadcast to (B,T,D)
        return self.dropout(x)


class MazeTransformer(nn.Module):
    def __init__(self, vocab_size, pad_idx):
        super().__init__()

        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, TR_D_MODEL)
        self.pos_encoder = SinusoidalPositionalEncoding(TR_D_MODEL, TR_DROPOUT)

        self.transformer = nn.Transformer(
            d_model=TR_D_MODEL,
            nhead=TR_NHEAD,
            num_encoder_layers=TR_NUM_ENCODER_LAYERS,
            num_decoder_layers=TR_NUM_DECODER_LAYERS,
            dim_feedforward=TR_DIM_FEEDFORWARD,
            dropout=TR_DROPOUT,
            batch_first=True
        )

        self.fc_out = nn.Linear(TR_D_MODEL, vocab_size)
        # weight tying
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
        # src, tgt: (B,T)
        src_key_padding_mask = (src == self.pad_idx).to(DEVICE)
        tgt_key_padding_mask = (tgt == self.pad_idx).to(DEVICE)
        tgt_causal_mask = self._generate_square_subsequent_mask(tgt.size(1))

        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(TR_D_MODEL))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(TR_D_MODEL))

        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_causal_mask
        )
        return self.fc_out(output)


# =====================================================
# 3. MODEL BUILDERS
# =====================================================

def build_rnn_model(vocab_size, pad_idx):
    attn = Attention(
        encoder_hidden_dim=RNN_HIDDEN_DIM,
        decoder_hidden_dim=RNN_HIDDEN_DIM
    )
    enc = EncoderRNN(
        input_vocab_size=vocab_size,
        embedding_dim=RNN_EMBEDDING_DIM,
        hidden_dim=RNN_HIDDEN_DIM,
        n_layers=RNN_N_LAYERS,
        dropout=RNN_DROPOUT,
        pad_idx=pad_idx
    )
    dec = DecoderRNN(
        output_vocab_size=vocab_size,
        embedding_dim=RNN_EMBEDDING_DIM,
        encoder_hidden_dim=RNN_HIDDEN_DIM,
        decoder_hidden_dim=RNN_HIDDEN_DIM,
        n_layers=RNN_N_LAYERS,
        dropout=RNN_DROPOUT,
        attention=attn,
        pad_idx=pad_idx
    )
    return Seq2Seq(enc, dec, DEVICE).to(DEVICE)


def build_transformer_model(vocab_size, pad_idx):
    return MazeTransformer(vocab_size, pad_idx).to(DEVICE)


# =====================================================
# 4. ENCODING / DECODING HELPERS
# =====================================================

def encode_input_sequence(input_seq_str, token_to_idx, flip=False):
    """
    input_seq_str: string like "['<ADJLIST START>', ...]"
    flip=True for RNN (because you reversed in training).
    """
    tokens = ast.literal_eval(input_seq_str)
    indices = [token_to_idx[t] for t in tokens]
    src = torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # [1, T]
    if flip:
        src = src.flip(dims=[1])
    return src.to(DEVICE)


def predict_path_rnn(model, src_tensor, sos_idx, eos_idx, idx_to_token, max_len=30):
    """
    Greedy decoding for RNN model. src_tensor: [1, src_len]
    Returns list of tokens, including <SOS>, <EOS>.
    """
    model.eval()
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        trg_indices = [sos_idx]

        for _ in range(max_len):
            last = torch.LongTensor([trg_indices[-1]]).to(DEVICE)
            output, hidden, _ = model.decoder(last, hidden, encoder_outputs)
            pred_token = output.argmax(1).item()
            trg_indices.append(pred_token)
            if pred_token == eos_idx:
                break

    return [idx_to_token[i] for i in trg_indices]


def predict_path_transformer(model, src_tensor, sos_idx, eos_idx, idx_to_token, max_len=50):
    """
    Greedy decoding for Transformer model (MazeTransformer).
    src_tensor: [1, src_len]
    """
    model.eval()
    with torch.no_grad():
        # Encode source
        src_emb = model.pos_encoder(model.embedding(src_tensor) * math.sqrt(TR_D_MODEL))
        memory = model.transformer.encoder(src_emb)

        trg_indices = [sos_idx]
        for _ in range(max_len):
            tgt = torch.tensor([trg_indices], dtype=torch.long).to(DEVICE)
            tgt_emb = model.pos_encoder(model.embedding(tgt) * math.sqrt(TR_D_MODEL))
            tgt_mask = model._generate_square_subsequent_mask(len(trg_indices))

            out = model.transformer.decoder(
                tgt_emb, memory, tgt_mask=tgt_mask
            )
            next_tok = model.fc_out(out[:, -1]).argmax(-1).item()
            trg_indices.append(next_tok)
            if next_tok == eos_idx:
                break

    return [idx_to_token[i] for i in trg_indices]


def clean_output_tokens(tokens):
    return [t for t in tokens if t not in [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]]


# =====================================================
# 5. MAIN EVAL LOGIC
# =====================================================

def main():
    if len(sys.argv) != 5:
        print(
            "Usage: python eval.py <model_path> <model_type> <data_csv_path> <output_csv_path>\n"
            "  <model_type> must be 'rnn' or 'transformer'."
        )
        sys.exit(1)

    model_path = sys.argv[1]
    model_type = sys.argv[2].lower()
    data_csv_path = sys.argv[3]
    output_csv_path = sys.argv[4]

    if model_type not in ["rnn", "transformer"]:
        print("Error: model_type must be 'rnn' or 'transformer'.")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Error: model file not found at {model_path}")
        sys.exit(1)

    if not os.path.exists(data_csv_path):
        print(f"Error: data csv file not found at {data_csv_path}")
        sys.exit(1)

    # -------- SELECT VOCAB BASED ON MODEL TYPE --------
    if model_type == "rnn":
        token_to_idx = VOCAB_RNN
    else:
        token_to_idx = VOCAB_TRANSFORMER

    if not token_to_idx:
        raise RuntimeError(
            f"The vocabulary for {model_type} is empty. "
            f"Please paste your vocab.json content into VOCAB_{model_type.upper()}."
        )

    idx_to_token = {idx: tok for tok, idx in token_to_idx.items()}
    pad_idx = token_to_idx[PAD_TOKEN]
    sos_idx = token_to_idx[SOS_TOKEN]
    eos_idx = token_to_idx[EOS_TOKEN]
    vocab_size = len(token_to_idx)

    # -------- BUILD MODEL AND LOAD WEIGHTS --------
    if model_type == "rnn":
        model = build_rnn_model(vocab_size, pad_idx)
    else:
        model = build_transformer_model(vocab_size, pad_idx)

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # -------- READ INPUT CSV --------
    df = pd.read_csv(data_csv_path)

    # Find the input sequence column
    if "input_sequence" in df.columns:
        input_col = "input_sequence"
    elif "inputs" in df.columns:
        input_col = "inputs"
    else:
        raise Exception(
            f"Could not find input column. Available columns: {list(df.columns)}"
        )

    # Ensure an id column exists
    if "id" not in df.columns:
        df["id"] = range(len(df))

    predicted_paths = []

    # -------- PREDICT ROW BY ROW --------
    for _, row in df.iterrows():
        input_seq_str = row[input_col]
        flip = (model_type == "rnn")  # RNN was trained with reversed input
        src_tensor = encode_input_sequence(input_seq_str, token_to_idx, flip=flip)

        if model_type == "rnn":
            decoded_tokens = predict_path_rnn(
                model, src_tensor, sos_idx, eos_idx, idx_to_token, max_len=30
            )
        else:
            decoded_tokens = predict_path_transformer(
                model, src_tensor, sos_idx, eos_idx, idx_to_token, max_len=50
            )

        clean_tokens = clean_output_tokens(decoded_tokens)
        predicted_paths.append(str(clean_tokens))

    # -------- BUILD OUTPUT DF --------
    out_df = df.copy()
    out_df["predicted_path"] = predicted_paths

    # If your sample_submission has only ['id','predicted_path'], uncomment:
    # out_df = out_df[["id", "predicted_path"]]

    # -------- SAVE --------
    out_dir = os.path.dirname(os.path.abspath(output_csv_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    out_df.to_csv(output_csv_path, index=False)
    print(f"Saved predictions to: {output_csv_path}")


if __name__ == "__main__":
    main()
