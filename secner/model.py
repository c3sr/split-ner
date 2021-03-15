import torch
import torch.nn as nn
from secner.additional_args import AdditionalArguments
from secner.cnn import CharCNN
from secner.dataset import NerDataset
from secner.flair_cnn import FlairCNN
from secner.loss import DiceLoss, CrossEntropyPunctuationLoss
from transformers import BertConfig
from transformers.models.bert import BertModel, BertPreTrainedModel


class NerModel(BertPreTrainedModel):

    def __init__(self, config: BertConfig, additional_args: AdditionalArguments):
        super(NerModel, self).__init__(config)
        self.additional_args = additional_args
        self.num_labels = config.num_labels
        self.num_word_types = len(NerDataset.get_word_type_vocab())
        none_tag = self.additional_args.none_tag
        self.num_pos_tags = len(NerDataset.parse_aux_tag_vocab(self.additional_args.pos_tag_vocab_path, none_tag,
                                                               self.additional_args.use_pos_tag))
        self.num_dep_tags = len(NerDataset.parse_aux_tag_vocab(self.additional_args.dep_tag_vocab_path, none_tag,
                                                               self.additional_args.use_dep_tag))
        self.ignore_label = nn.CrossEntropyLoss().ignore_index

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        classifier_inp_dim = self.bert.config.hidden_size

        if self.additional_args.word_type_handling == "1hot":
            classifier_inp_dim += self.num_word_types

        if self.additional_args.use_pos_tag:
            classifier_inp_dim += self.num_pos_tags

        if self.additional_args.use_dep_tag:
            classifier_inp_dim += self.num_dep_tags

        if self.additional_args.punctuation_handling != "none":
            self.punctuation_vocab_size = NerDataset.get_punctuation_vocab_size(
                self.additional_args.punctuation_handling)
            classifier_inp_dim += self.punctuation_vocab_size

        if self.additional_args.gold_span_inp == "simple":
            classifier_inp_dim += 1
        elif self.additional_args.gold_span_inp == "label":
            classifier_inp_dim += self.num_labels

        if self.additional_args.use_char_cnn in ["char", "both"]:
            self.char_cnn = CharCNN(additional_args, "char")
            classifier_inp_dim += self.char_cnn.char_out_dim

        if self.additional_args.use_char_cnn in ["flair", "both-flair"]:
            self.flair_cnn = FlairCNN(additional_args)
            classifier_inp_dim += self.flair_cnn.out_dim

        if self.additional_args.use_char_cnn in ["pattern", "both", "both-flair"]:
            dropout_prob = config.hidden_dropout_prob if self.additional_args.lstm_num_layers > 1 else 0.
            self.pattern_cnn = CharCNN(additional_args, "pattern")
            self.pattern_lstm = nn.LSTM(input_size=self.pattern_cnn.char_out_dim,
                                        hidden_size=self.additional_args.lstm_hidden_dim,
                                        bidirectional=True,
                                        batch_first=True,
                                        num_layers=self.additional_args.lstm_num_layers,
                                        dropout=dropout_prob)
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
            head_mask=None,
            char_ids=None,
            pattern_ids=None,
            flair_ids=None,
            flair_attention_mask=None,
            flair_boundary=None,
            punctuation_vec=None,
            gold_span_inp=None,
            word_type_ids=None,
            pos_tag=None,
            dep_tag=None,
            labels=None,
            **kwargs):

        batch_size, seq_len = input_ids.shape
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]

        sequence_output = self.compress_with_head_mask(head_mask, sequence_output, 0.0)
        attention_mask = self.compress_with_head_mask(head_mask, attention_mask, 0)

        if self.additional_args.punctuation_handling == "type1":
            sequence_output = torch.cat([sequence_output, punctuation_vec.unsqueeze(-1)], dim=2)
        elif self.additional_args.punctuation_handling == "type1-and":
            vec = NerModel.expand_punctuation_vec(punctuation_vec)
            sequence_output = torch.cat([sequence_output, vec], dim=2)
        elif self.additional_args.punctuation_handling == "type2":
            punctuation_one_hot_vec = torch.eye(self.punctuation_vocab_size)[punctuation_vec].to(sequence_output.device)
            sequence_output = torch.cat([sequence_output, punctuation_one_hot_vec], dim=2)

        if self.additional_args.word_type_handling == "1hot":
            word_type_vec = torch.eye(self.num_word_types)[word_type_ids].to(sequence_output.device)
            sequence_output = torch.cat([sequence_output, word_type_vec], dim=2)

        if self.additional_args.use_pos_tag:
            pos_tag_vec = torch.eye(self.num_pos_tags)[pos_tag].to(sequence_output.device)
            sequence_output = torch.cat([sequence_output, pos_tag_vec], dim=2)

        if self.additional_args.use_dep_tag:
            dep_tag_vec = torch.eye(self.num_dep_tags)[dep_tag].to(sequence_output.device)
            sequence_output = torch.cat([sequence_output, dep_tag_vec], dim=2)

        if self.additional_args.gold_span_inp == "simple":
            sequence_output = torch.cat([sequence_output, gold_span_inp.unsqueeze(-1)], dim=2)

        elif self.additional_args.gold_span_inp == "label":
            span_vec = torch.eye(self.num_labels)[gold_span_inp].to(sequence_output.device)
            sequence_output = torch.cat([sequence_output, span_vec], dim=2)

        if self.additional_args.use_char_cnn in ["char", "both"]:
            char_vec = self.char_cnn(char_ids)
            sequence_output = torch.cat([sequence_output, char_vec], dim=2)

        if self.additional_args.use_char_cnn in ["flair", "both-flair"]:
            flair_vec = self.flair_cnn(flair_ids, flair_attention_mask, flair_boundary)
            sequence_output = torch.cat([sequence_output, flair_vec], dim=2)

        if self.additional_args.use_char_cnn in ["pattern", "both", "both-flair"]:
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

        predictions = torch.argmax(logits, dim=2)
        outputs = (predictions,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1).eq(1)
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(self.ignore_label).type_as(labels)
                )
            else:
                active_logits = logits.view(-1, self.num_labels)
                active_labels = labels.view(-1)

            if self.additional_args.loss_type == "dice":
                loss = DiceLoss()(active_logits, active_labels, attention_mask.view(-1))
            elif self.additional_args.loss_type == "ce_wt":
                loss = nn.CrossEntropyLoss(weight=self.loss_wt.to(active_logits.device))(active_logits, active_labels)
            elif self.additional_args.loss_type == "ce_punct":
                loss = CrossEntropyPunctuationLoss()(active_logits, active_labels, attention_mask.view(-1),
                                                     punctuation_vec.view(-1))
            else:
                loss = nn.CrossEntropyLoss()(active_logits, active_labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

    def compress_with_head_mask(self, head_mask, x, pad_value):
        if not self.additional_args.use_head_mask:
            return x
        new_x = torch.full(x.shape, fill_value=pad_value, dtype=x.dtype, device=x.device)
        for i in range(head_mask.shape[0]):
            k = 0
            for j in range(head_mask.shape[1]):
                if head_mask[i, j] == 1:
                    new_x[i, k] = x[i, j]
                    k += 1
        return new_x

    def expand_with_head_mask(self, head_mask, x, pad_value):
        if not self.additional_args.use_head_mask:
            return x
        new_x = torch.full(x.shape, fill_value=pad_value, dtype=x.dtype, device=x.device)
        for i in range(head_mask.shape[0]):
            k = -1
            for j in range(head_mask.shape[1]):
                if head_mask[i, j] == 1:
                    k += 1
                new_x[i, j] = x[i, k]
        return new_x

    @staticmethod
    def expand_punctuation_vec(punctuation_vec):
        vec = torch.zeros(punctuation_vec.shape[0], punctuation_vec.shape[1], 2, device=punctuation_vec.device)
        for i in range(punctuation_vec.shape[0]):
            for j in range(punctuation_vec.shape[1]):
                if punctuation_vec[i, j] != -1:
                    vec[i, j, punctuation_vec[i, j]] = 1.0
        return vec
