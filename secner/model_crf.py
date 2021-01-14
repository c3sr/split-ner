import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
from torchcrf import CRF
from transformers import BertConfig
from transformers.models.bert import BertModel, BertPreTrainedModel


class NerModelWithCrf(BertPreTrainedModel):

    def __init__(self, config: BertConfig):
        super(NerModelWithCrf, self).__init__(config)
        self.num_labels = config.num_labels

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
        emissions = log_softmax(self.classifier(sequence_output), dim=-1)
        crf_attention_mask = attention_mask.type(torch.uint8) if torch.is_tensor(attention_mask) else None
        predictions = self.crf.decode(emissions, crf_attention_mask)
        padded_predictions = [p + [-100] * (input_ids.shape[1] - len(p)) for p in predictions]
        tag_seq = torch.Tensor(padded_predictions).to(dtype=torch.int64, device=input_ids.device)

        outputs = (tag_seq,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            crf_labels = labels.clone()
            # since negative idandices is not supported by the CRF library
            # (changing it to any positive index should have no effect)
            crf_labels[crf_labels == -100] = 0
            loss = -self.crf.forward(emissions, crf_labels, crf_attention_mask, reduction="mean")
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
