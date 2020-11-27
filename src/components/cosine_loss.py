import torch
from torch import nn as nn


class ModelLoss(nn.Module):

    def __init__(self, tag_emb, loss_type, device):
        super(ModelLoss, self).__init__()
        self.loss_type = loss_type

        if self.loss_type == "cosine":
            self.tag_emb = torch.as_tensor(tag_emb, device=device)
            self.cos = nn.CosineSimilarity(dim=3)
            self.loss = nn.MSELoss(reduction="sum")
        else:
            self.loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inp, mask, target_indices):
        if self.loss_type == "cosine":
            inp = inp.transpose(1, 2)
            batch_size, seq_len, emb_dim = inp.shape
            num_tags = self.tag_emb.shape[0]

            flattened_indices = target_indices.flatten()
            target = torch.index_select(self.tag_emb, dim=0, index=flattened_indices)
            target = target.reshape(batch_size, seq_len, emb_dim)

            inp = inp.unsqueeze(2).expand(batch_size, seq_len, num_tags, emb_dim)
            target = target.unsqueeze(2).expand(batch_size, seq_len, num_tags, emb_dim)

            tag_emb = self.tag_emb.unsqueeze(0).expand((seq_len, num_tags, emb_dim))
            tag_emb = tag_emb.unsqueeze(0).expand(batch_size, seq_len, num_tags, emb_dim)

            c_inp = (1.0 - self.cos(inp, tag_emb)).transpose(1, 2)
            c_target = (1.0 - self.cos(target, tag_emb)).transpose(1, 2)

            c_inp = c_inp * mask.unsqueeze(1).expand_as(c_inp)
            c_target = c_target * mask.unsqueeze(1).expand_as(c_target)

            loss = self.loss(c_inp, c_target)

        else:
            loss = torch.sum(self.loss(inp, target_indices) * mask)

        return loss
