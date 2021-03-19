import torch
import torch.nn as nn
from transformers import BertConfig
from transformers.models.bert import BertModel, BertPreTrainedModel

from secner.additional_args import AdditionalArguments


class NerSpanModel(BertPreTrainedModel):

    def __init__(self, config: BertConfig, additional_args: AdditionalArguments):
        super(NerSpanModel, self).__init__(config)
        self.additional_args = additional_args
        self.num_labels = config.num_labels
        self.ignore_label = nn.CrossEntropyLoss().ignore_index

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        classifier_inp_dim = self.bert.config.hidden_size

        if self.additional_args.second_classifier_hidden_sz > 0:
            self.hidden_classifier = nn.Linear(classifier_inp_dim, self.additional_args.second_classifier_hidden_sz)
            classifier_inp_dim = self.additional_args.second_classifier_hidden_sz

        self.classifier = nn.Linear(classifier_inp_dim, self.num_labels)

        # Downscaling contribution of "O" terms by fixed constant factor for now
        self.loss_wt = torch.tensor([1.0] * (self.num_labels - 1) + [0.5])

        self.init_weights()

        if self.additional_args.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            **kwargs):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs[1]

        if self.additional_args.second_classifier_hidden_sz > 0:
            pooled_output = self.dropout(pooled_output)
            pooled_output = self.hidden_classifier(pooled_output)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        predictions = torch.argmax(logits, dim=1)
        outputs = (predictions,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.additional_args.loss_type == "ce_wt":
                loss = nn.CrossEntropyLoss(weight=self.loss_wt.to(logits.device))(logits, labels)
            else:
                loss = nn.CrossEntropyLoss()(logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
