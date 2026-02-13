# model.py  (ChatGPT version – padding-aware, more numerically stable)

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class EncoderRNN(nn.Module):

    def _init_(self, input_vocab_size, embedding_dim, hidden_dim,
                 n_layers, dropout, pad_idx):
        super()._init_()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(
            num_embeddings=input_vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True
        )

    def forward(self, src_tensor):
        # [batch, src_len, emb]
        embedded = self.dropout(self.embedding(src_tensor))

        # lengths: number of non-pad tokens per sequence
        with torch.no_grad():
            src_lengths = (src_tensor != self.pad_idx).sum(dim=1)

        # pack -> RNN -> unpack
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths=src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        packed_outputs, hidden = self.rnn(packed)
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=src_tensor.size(1)
        )
        # encoder_outputs: [batch, src_len, hidden_dim]
        # hidden: [n_layers, batch, hidden_dim]
        return encoder_outputs, hidden


class Attention(nn.Module):

    def _init_(self, encoder_hidden_dim, decoder_hidden_dim):
        super()._init_()
        self.attn_enc = nn.Linear(encoder_hidden_dim, decoder_hidden_dim)
        self.attn_dec = nn.Linear(decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, src_mask=None):
        batch_size, src_len, _ = encoder_outputs.shape
        # use top-layer hidden state
        dec_hidden_top = decoder_hidden[-1]  # [batch, dec_hidden_dim]
        dec_hidden_repeated = dec_hidden_top.unsqueeze(1).repeat(1, src_len, 1)

        # energy: [batch, src_len, dec_hidden_dim]
        energy = torch.tanh(
            self.attn_enc(encoder_outputs) + self.attn_dec(dec_hidden_repeated)
        )
        # scores: [batch, src_len]
        scores = self.v(energy).squeeze(2)

        # mask PAD positions so they get almost zero attention weight
        if src_mask is not None:
            scores = scores.masked_fill(src_mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=1)  # [batch, src_len]

        # [batch, 1, src_len]
        weights_expanded = weights.unsqueeze(1)

        # [batch, 1, enc_hidden_dim] -> [batch, enc_hidden_dim]
        context = torch.bmm(weights_expanded, encoder_outputs).squeeze(1)
        return context, weights


class DecoderRNN(nn.Module):

    def _init_(self, output_vocab_size, embedding_dim,
                 encoder_hidden_dim, decoder_hidden_dim,
                 n_layers, dropout, attention, pad_idx):
        super()._init_()

        self.output_vocab_size = output_vocab_size
        self.attention = attention
        self.n_layers = n_layers
        self.decoder_hidden_dim = decoder_hidden_dim
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(
            num_embeddings=output_vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )
        self.dropout = nn.Dropout(dropout)

        # RNN input = [embedding | context]
        self.rnn = nn.RNN(
            input_size=embedding_dim + encoder_hidden_dim,
            hidden_size=decoder_hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc_out = nn.Linear(decoder_hidden_dim, output_vocab_size)

    def forward(self, dec_input_token, dec_hidden, encoder_outputs, src_mask=None):
        # [batch, 1]
        dec_input_token = dec_input_token.unsqueeze(1)

        # [batch, 1, emb]
        embedded = self.dropout(self.embedding(dec_input_token))

        # context: [batch, enc_hidden_dim]
        context, attn_weights = self.attention(dec_hidden, encoder_outputs, src_mask=src_mask)
        context = context.unsqueeze(1)  # [batch, 1, enc_hidden_dim]

        # [batch, 1, emb + enc_hidden_dim]
        rnn_input = torch.cat((embedded, context), dim=2)

        # [batch, 1, dec_hidden_dim]
        dec_output, dec_hidden = self.rnn(rnn_input, dec_hidden)

        prediction = self.fc_out(dec_output.squeeze(1))  # [batch, vocab_size]
        return prediction, dec_hidden, attn_weights


class Seq2Seq(nn.Module):
    def _init_(self, encoder, decoder, pad_idx, device):
        super()._init_()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device

    def make_src_mask(self, src_tensor):
        return (src_tensor != self.pad_idx).long()

    def forward(self, src_tensor, trg_tensor, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg_tensor.shape
        vocab_size = self.decoder.output_vocab_size

        # Encode
        encoder_outputs, hidden = self.encoder(src_tensor)
        src_mask = self.make_src_mask(src_tensor)  # [batch, src_len]

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, vocab_size, device=self.device)

        # First decoder input is <SOS>
        dec_input = trg_tensor[:, 0]  # [batch]

        for t in range(1, trg_len):
            dec_output, hidden, _ = self.decoder(
                dec_input, hidden, encoder_outputs, src_mask=src_mask
            )
            outputs[:, t] = dec_output

            use_teacher_force = (random.random() < teacher_forcing_ratio)
            top1 = dec_output.argmax(1)  # [batch]

            dec_input = trg_tensor[:, t] if use_teacher_force else top1

        return outputs