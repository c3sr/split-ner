import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class NerModel(BertPreTrainedModel):

    def __init__(self, config, ner_params):
        super(NerModel, self).__init__(config)
        self.ner_params = ner_params
        self.num_tags = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.fc = nn.Linear(self.bert.config.hidden_size, config.num_labels)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.criterion = nn.CrossEntropyLoss()
        self.init_weights()

    def forward(self, token_ids, attention_mask=None, tag_ids=None):
        token_ids, attention_mask, tag_ids = self.push_to_device(token_ids, attention_mask, tag_ids)
        bert_outputs = self.bert(token_ids, attention_mask=attention_mask)
        output = self.dropout(bert_outputs[0])
        logits = self.fc(output)
        if tag_ids is None:
            # evaluation flow
            return torch.argmax(logits, dim=-1)
        # training flow
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_tags)
            active_labels = torch.where(
                active_loss, tag_ids.view(-1), torch.tensor(self.criterion.ignore_index).type_as(tag_ids)
            )
            return self.criterion(active_logits, active_labels)
        return self.criterion(logits.view(-1, self.num_tags), tag_ids.view(-1))

    def push_to_device(self, token_ids, attention_mask=None, tag_ids=None):
        token_ids = token_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if tag_ids is not None:
            tag_ids = tag_ids.to(self.device)
        return token_ids, attention_mask, tag_ids
