import torch
import torch.nn as nn
from secner.additional_args import AdditionalArguments
from secner.cnn import CharCNN
from secner.loss import DiceLoss
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

        if self.additional_args.punctuation_handling:
            classifier_inp_dim += 1

        if self.additional_args.use_char_cnn in ["char", "both"]:
            self.char_cnn = CharCNN(additional_args)
            classifier_inp_dim += self.char_cnn.char_out_dim

        if self.additional_args.use_char_cnn in ["pattern", "both"]:
            self.pattern_cnn = CharCNN(additional_args)
            self.pattern_lstm = nn.LSTM(input_size=self.pattern_cnn.char_out_dim,
                                        hidden_size=self.additional_args.lstm_hidden_dim,
                                        bidirectional=True,
                                        batch_first=True,
                                        num_layers=1)
            classifier_inp_dim += 2 * self.additional_args.lstm_hidden_dim

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

        # Downscaling contribution of "O" terms by fixed constant factor for now
        self.loss_wt = torch.tensor([1.0] * (self.num_labels - 1) + [0.5])

        if self.additional_args.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            char_ids=None,
            pattern_ids=None,
            punctuation_vec=None,
            labels=None,
            **kwargs):

        batch_size, seq_len = input_ids.shape
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        if self.additional_args.punctuation_handling:
            sequence_output = torch.cat([sequence_output, punctuation_vec.unsqueeze(-1)], dim=2)

        if self.additional_args.use_char_cnn in ["char", "both"]:
            char_vec = self.char_cnn(char_ids)
            sequence_output = torch.cat([sequence_output, char_vec], dim=2)

        if self.additional_args.use_char_cnn in ["pattern", "both"]:
            pattern_vec = self.pattern_cnn(pattern_ids)
            lengths = torch.as_tensor(attention_mask.sum(1).int(), dtype=torch.int64, device=torch.device("cpu"))
            packed_inp = nn.utils.rnn.pack_padded_sequence(input=pattern_vec,
                                                           lengths=lengths,
                                                           batch_first=True,
                                                           enforce_sorted=False)
            self.pattern_lstm.flatten_parameters()
            packed_out, _ = self.pattern_lstm(packed_inp)
            pattern_vec, _ = nn.utils.rnn.pad_packed_sequence(sequence=packed_out,
                                                              batch_first=True,
                                                              total_length=seq_len)
            pattern_vec = self.dropout(pattern_vec)
            sequence_output = torch.cat([sequence_output, pattern_vec], dim=2)

        if self.additional_args.use_end_cnn:
            sequence_output = self.end_cnn(sequence_output.unsqueeze(1))
            sequence_output = sequence_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (torch.argmax(logits, dim=2),) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(nn.CrossEntropyLoss().ignore_index).type_as(labels)
                )
            else:
                active_logits = logits.view(-1, self.num_labels)
                active_labels = labels.view(-1)

            if self.additional_args.loss_type == "dice":
                loss = DiceLoss()(active_logits, active_labels, attention_mask.view(-1))
            elif self.additional_args.loss_type == "ce_wt":
                loss = nn.CrossEntropyLoss(weight=self.loss_wt.to(active_logits.device))(active_logits, active_labels)
            else:
                loss = nn.CrossEntropyLoss()(active_logits, active_labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
