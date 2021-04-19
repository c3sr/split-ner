import torch
import torch.nn as nn

from secner.additional_args import AdditionalArguments
from secner.dataset import NerDataset


class FlairCNN(nn.Module):

    def __init__(self, args: AdditionalArguments):
        super(FlairCNN, self).__init__()
        self.args = args

        # +3 ([START], [END], [PAD])
        num_embeddings = len(NerDataset.get_flair_vocab()) + 3
        self.out_dim = self.args.lstm_hidden_dim * 4
        self.emb = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=self.args.char_emb_dim)
        self.lstm = nn.LSTM(input_size=self.args.char_emb_dim,
                            hidden_size=self.args.lstm_hidden_dim,
                            bidirectional=True,
                            batch_first=True,
                            num_layers=self.args.lstm_num_layers)

    def forward(self, flair_ids, flair_attention_mask, flair_boundary):
        x = self.emb(flair_ids)
        lengths = torch.as_tensor(flair_attention_mask.sum(1).int(), dtype=torch.int64, device=torch.device("cpu"))
        packed_inp = nn.utils.rnn.pack_padded_sequence(input=x,
                                                       lengths=lengths,
                                                       batch_first=True,
                                                       enforce_sorted=False)
        self.lstm.flatten_parameters()
        packed_out, _ = self.lstm(packed_inp)
        out, _ = nn.utils.rnn.pad_packed_sequence(sequence=packed_out, batch_first=True, total_length=x.shape[1])
        word_emb = torch.zeros(flair_boundary.shape[0], flair_boundary.shape[1] - 1, self.out_dim,
                               dtype=torch.int64,
                               device=flair_boundary.device)
        for b in range(flair_boundary.shape[0]):
            for s in range(1, flair_boundary.shape[1]):
                i, j = flair_boundary[b, s - 1].item(), flair_boundary[b, s].item()
                if i != -1 and j != -1:
                    word_emb[b, s - 1] = torch.cat([out[b, i, :], out[b, j, :]])
        return word_emb
