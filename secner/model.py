import torch
import torch.nn as nn
from secner.additional_args import AdditionalArguments
from secner.cnn import CharCNN
from transformers import BertConfig
from transformers.models.bert import BertModel, BertPreTrainedModel


class NerModel(BertPreTrainedModel):

    def __init__(self, config: BertConfig, additional_args: AdditionalArguments):
        super(NerModel, self).__init__(config)
        self.additional_args = additional_args
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        classifier_inp_dim = self.bert.config.hidden_size

        if self.additional_args.use_char_cnn:
            self.char_cnn = CharCNN(additional_args)
            classifier_inp_dim += self.char_cnn.char_out_dim

        if self.additional_args.use_end_cnn:
            self.end_cnn = nn.Conv2d(in_channels=1,
                                     out_channels=self.additional_args.end_cnn_channels,
                                     kernel_size=(5, 5),
                                     stride=(1, 2),
                                     padding=2,
                                     padding_mode="circular")
            classifier_inp_dim *= self.additional_args.end_cnn_channels // 2

        self.classifier = nn.Linear(classifier_inp_dim, self.num_labels)

        self.init_weights()

        if self.additional_args.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            char_ids=None,
            labels=None,
            **kwargs):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]

        if self.additional_args.use_char_cnn:
            char_vec = self.char_cnn(char_ids)
            sequence_output = torch.cat([sequence_output, char_vec], dim=2)

        if self.additional_args.use_end_cnn:
            batch_size, seq_len = input_ids.shape
            sequence_output = self.end_cnn(sequence_output.unsqueeze(1))
            sequence_output = sequence_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
