import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
from torchcrf import CRF
from transformers import BertConfig
from transformers.models.bert import BertModel, BertPreTrainedModel


class NerModelWithCrf(BertPreTrainedModel):

    def __init__(self, config: BertConfig):
        super(NerModelWithCrf, self).__init__(config)
        # self.num_labels = config.num_labels
        self.num_labels = config.num_labels + 1

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

        # TODO: arorja: check if different param initialization for CRF reqd.?
        self.init_weights()

        # for param in self.bert.parameters():
        #     param.requires_grad = False

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)
        attention_mask = attention_mask.type(torch.uint8) if attention_mask else None
        predictions = self.crf.decode(log_softmax(emissions, dim=-1), attention_mask)
        tag_seq = torch.Tensor([p + [-100] * (input_ids.shape[1] - len(p)) for p in predictions]).type(torch.int64)

        outputs = (tag_seq,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss = -self.crf.forward(log_softmax(emissions, dim=-1), labels, attention_mask, reduction="mean")
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
