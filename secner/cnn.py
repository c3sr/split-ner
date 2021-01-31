import torch
import torch.nn as nn

from secner.dataset import NerDataset


class CharCNN(nn.Module):

    def __init__(self, args):
        super(CharCNN, self).__init__()
        self.args = args
        num_embeddings = len(NerDataset.get_char_vocab()) + 1  # +1 (for the special [PAD] char)
        self.char_out_dim = self.args.char_emb_dim * self.args.cnn_num_filters

        self.emb = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=self.args.char_emb_dim)
        self.dropout = nn.Dropout(p=self.args.cnn_dropout_rate)
        self.cnn = nn.Conv1d(in_channels=self.args.char_emb_dim,
                             out_channels=self.char_out_dim,
                             kernel_size=self.args.cnn_kernel_size,
                             groups=self.args.char_emb_dim)

    def forward(self, char_ids):
        x = self.emb(char_ids)
        x = self.dropout(x)
        batch_size, seq_len, word_len, emb_dim = x.shape
        x = x.permute(0, 1, 3, 2)
        out = torch.zeros(batch_size, seq_len, self.char_out_dim, device=x.device, dtype=x.dtype)
        for k in range(seq_len):
            out[:, k, :], _ = torch.max(self.cnn(x[:, k, :, :]), dim=2)
        return out
