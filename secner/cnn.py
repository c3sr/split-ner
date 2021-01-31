import torch
import torch.nn as nn
import torch.nn.functional as F

from secner.dataset import NerDataset


class CharCNN(nn.Module):

    def __init__(self, args):
        super(CharCNN, self).__init__()
        self.args = args
        num_embeddings = len(NerDataset.get_char_vocab()) + 1  # +1 (for the special [PAD] char)
        self.cnn_out_dim = self.args.char_emb_dim * self.args.cnn_num_filters
        self.char_out_dim = 768

        self.emb = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=self.args.char_emb_dim)
        self.dropout = nn.Dropout(p=self.args.cnn_dropout_rate)
        self.cnn1 = nn.Conv1d(in_channels=self.args.char_emb_dim,
                              out_channels=self.cnn_out_dim,
                              kernel_size=1,
                              groups=self.args.char_emb_dim)
        self.cnn2 = nn.Conv1d(in_channels=self.args.char_emb_dim,
                              out_channels=self.cnn_out_dim,
                              kernel_size=2,
                              groups=self.args.char_emb_dim)
        self.cnn3 = nn.Conv1d(in_channels=self.args.char_emb_dim,
                              out_channels=self.cnn_out_dim,
                              kernel_size=3,
                              groups=self.args.char_emb_dim)
        self.cnn4 = nn.Conv1d(in_channels=self.args.char_emb_dim,
                              out_channels=self.cnn_out_dim,
                              kernel_size=4,
                              groups=self.args.char_emb_dim)
        self.cnn5 = nn.Conv1d(in_channels=self.args.char_emb_dim,
                              out_channels=self.cnn_out_dim,
                              kernel_size=5,
                              groups=self.args.char_emb_dim)
        self.lin = nn.Linear(self.cnn_out_dim * 5, self.char_out_dim)

    def forward(self, char_ids):
        x = self.emb(char_ids)
        # x = self.dropout(x)
        batch_size, seq_len, word_len, emb_dim = x.shape
        x = x.permute(0, 1, 3, 2)
        out = torch.zeros(batch_size, seq_len, 5 * self.cnn_out_dim, device=x.device, dtype=x.dtype)
        for k in range(seq_len):
            x_proj = x[:, k, :, :]
            v1, _ = torch.max(self.cnn1(x_proj), dim=2)
            v2, _ = torch.max(self.cnn2(x_proj), dim=2)
            v3, _ = torch.max(self.cnn3(x_proj), dim=2)
            v4, _ = torch.max(self.cnn4(x_proj), dim=2)
            v5, _ = torch.max(self.cnn5(x_proj), dim=2)
            out[:, k, :] = F.relu(torch.cat([v1, v2, v3, v4, v5], dim=1))
        out = self.lin(out)
        return out
