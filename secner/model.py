import torch
import torch.nn as nn
from transformers import BertModel


class NerModel(nn.Module):

    def __init__(self, config, device):
        super(NerModel, self).__init__()
        self.config = config
        self.device = device
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert_model.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(self.bert_model.config.hidden_size, 34)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, token_ids):
        token_ids = token_ids.to(self.device)
        x = self.bert_model(token_ids)
        x = self.fc(x[0])
        return x

    def forward_train(self, token_ids, tag_ids):
        token_ids = token_ids.to(self.device)
        tag_ids = tag_ids.to(self.device)
        out = self(token_ids)
        return self.criterion(out.permute(0, 2, 1), tag_ids)

    def forward_eval(self, token_ids):
        token_ids = token_ids.to(self.device)
        token_ids = self(token_ids)
        return torch.argmax(token_ids, dim=-1)
