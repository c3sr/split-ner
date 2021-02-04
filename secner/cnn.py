import torch
import torch.nn as nn
import torch.nn.functional as F

from secner.dataset import NerDataset


class CharCNN(nn.Module):

    def __init__(self, args):
        super(CharCNN, self).__init__()
        self.args = args
        num_embeddings = len(NerDataset.get_char_vocab()) + 1  # +1 (for the special [PAD] char)
        self.char_out_dim = 768

        self.emb = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=self.args.char_emb_dim)
        self.dropout = nn.Dropout(p=self.args.cnn_dropout_rate)

        # found to work well
        self.cnn_layer_config = [[1, 16], [2, 16], [3, 16], [4, 16], [5, 16]]

        # self.cnn_layer_config = [[1, 16], [2, 16], [3, 16], [4, 16], [5, 16], [6, 16], [7, 16]]

        # not working as good as the above one
        # self.cnn_layer_config = [[1, 16], [2, 16], [3, 16], [4, 16], [5, 32], [6, 32], [7, 32], [8, 32], [9, 32]]

        self.hidden_dim = 0
        for i, (kernel_size, num_filters) in enumerate(self.cnn_layer_config):
            cnn_out_dim = self.args.char_emb_dim * num_filters
            conv = nn.Conv1d(in_channels=self.args.char_emb_dim,
                             out_channels=cnn_out_dim,
                             kernel_size=kernel_size,
                             groups=self.args.char_emb_dim)
            self.hidden_dim += cnn_out_dim
            self.add_module("char_conv_{}".format(i), conv)

        self.lin = nn.Linear(self.hidden_dim, self.char_out_dim)

    def forward(self, char_ids):
        x = self.emb(char_ids)
        # x = self.dropout(x)
        batch_size, seq_len, word_len, emb_dim = x.shape
        x = x.permute(0, 1, 3, 2)
        out = torch.zeros(batch_size, seq_len, self.hidden_dim, device=x.device, dtype=x.dtype)
        for k in range(seq_len):
            x_proj = x[:, k, :, :]
            cnn_outputs = []
            for i in range(len(self.cnn_layer_config)):
                conv = getattr(self, "char_conv_{}".format(i))
                v, _ = torch.max(conv(x_proj), dim=2)
                cnn_outputs.append(v)
            out[:, k, :] = F.relu(torch.cat(cnn_outputs, dim=1))
        out = self.lin(out)
        return out
