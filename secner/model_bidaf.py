import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig
from transformers.models.bert import BertModel, BertPreTrainedModel

from secner.additional_args import AdditionalArguments
from secner.cnn import CharCNN
from secner.dataset import NerDataset
from secner.loss import DiceLoss, CrossEntropyPunctuationLoss
from secner.model import NerModel


class NerModelBiDAF(BertPreTrainedModel):

    def __init__(self, config: BertConfig, additional_args: AdditionalArguments):
        super(NerModelBiDAF, self).__init__(config)
        self.additional_args = additional_args
        self.num_labels = config.num_labels
        self.num_word_types = len(NerDataset.get_word_type_vocab())
        none_tag = self.additional_args.none_tag
        dropout_prob = config.hidden_dropout_prob if self.additional_args.lstm_num_layers > 1 else 0.
        self.num_pos_tags = len(NerDataset.parse_aux_tag_vocab(self.additional_args.pos_tag_vocab_path, none_tag,
                                                               self.additional_args.use_pos_tag))
        self.num_dep_tags = len(NerDataset.parse_aux_tag_vocab(self.additional_args.dep_tag_vocab_path, none_tag,
                                                               self.additional_args.use_dep_tag))
        self.ignore_label = nn.CrossEntropyLoss().ignore_index

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        classifier_inp_dim = 0

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

        char_vec_dim = 0
        if self.additional_args.use_bidaf_orig_cnn:
            # Character Embedding Layer
            num_embeddings = len(NerDataset.get_char_vocab()) + 1  # +1 (for the special [PAD] char)
            self.char_emb = nn.Embedding(num_embeddings=num_embeddings,
                                         embedding_dim=self.additional_args.char_emb_dim,
                                         padding_idx=0)
            # nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)

            self.char_conv = nn.Sequential(
                nn.Conv2d(in_channels=1,
                          out_channels=self.additional_args.cnn_num_filters,
                          kernel_size=(self.additional_args.char_emb_dim, self.additional_args.cnn_kernel_size)),
                nn.ReLU()
            )
            char_vec_dim += self.additional_args.cnn_num_filters
        else:
            if self.additional_args.use_char_cnn in ["char", "both"]:
                self.char_cnn = CharCNN(additional_args, "char")
                char_vec_dim += self.char_cnn.char_out_dim

            if self.additional_args.use_char_cnn in ["pattern", "both"]:
                self.pattern_cnn = CharCNN(additional_args, "pattern")
                self.pattern_lstm = nn.LSTM(input_size=self.pattern_cnn.char_out_dim,
                                            hidden_size=self.additional_args.lstm_hidden_dim,
                                            bidirectional=True,
                                            batch_first=True,
                                            num_layers=self.additional_args.lstm_num_layers,
                                            dropout=dropout_prob)
                char_vec_dim += 2 * self.additional_args.lstm_hidden_dim

        # highway network
        self.hidden_size = char_vec_dim + config.hidden_size
        for i in range(2):
            setattr(self, 'highway_linear{}'.format(i),
                    nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                  self.dropout,
                                  nn.ReLU()))
            setattr(self, 'highway_gate{}'.format(i),
                    nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                  self.dropout,
                                  nn.Sigmoid()))

        classifier_inp_dim += self.hidden_size
        self.classifier = nn.Linear(classifier_inp_dim, self.num_labels)

        self.init_weights()

        # Downscaling contribution of "O" terms by fixed constant factor for now
        self.loss_wt = torch.tensor([1.0] * (self.num_labels - 1) + [0.5])

        if self.additional_args.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def char_emb_layer(self, x):
        """
        :param x: (batch, seq_len, word_len)
        :return: (batch, seq_len, char_channel_size)
        """
        batch_size = x.size(0)
        # (batch, seq_len, word_len, char_dim)
        x = self.dropout(self.char_emb(x))
        # (batch， seq_len, char_dim, word_len)
        x = x.transpose(2, 3)
        # (batch * seq_len, 1, char_dim, word_len)
        x = x.view(-1, self.additional_args.char_emb_dim, x.size(3)).unsqueeze(1)
        # (batch * seq_len, char_channel_size, 1, conv_len) -> (batch * seq_len, char_channel_size, conv_len)
        x = self.char_conv(x).squeeze()
        # (batch * seq_len, char_channel_size, 1) -> (batch * seq_len, char_channel_size)
        x = F.max_pool1d(x, x.size(2)).squeeze()
        # (batch, seq_len, char_channel_size)
        x = x.view(batch_size, -1, self.additional_args.cnn_num_filters)

        return x

    def highway_network(self, x1, x2):
        """
        :param x1: (batch, seq_len, char_channel_size)
        :param x2: (batch, seq_len, word_dim)
        :return: (batch, seq_len, hidden_size)
        """
        # (batch, seq_len, char_channel_size + word_dim)
        x = torch.cat([x1, x2], dim=-1)
        for i in range(2):
            h = getattr(self, 'highway_linear{}'.format(i))(x)
            g = getattr(self, 'highway_gate{}'.format(i))(x)
            x = g * h + (1 - g) * x
        # (batch, seq_len, hidden_size)
        return x

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            char_ids=None,
            pattern_ids=None,
            punctuation_vec=None,
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

        if self.additional_args.use_bidaf_orig_cnn:
            char_vec = self.char_emb_layer(char_ids)
            sequence_output = self.highway_network(sequence_output, char_vec)
        else:
            additional_vec = torch.empty(batch_size, seq_len, 0, device=input_ids.device)
            if self.additional_args.use_char_cnn in ["char", "both"]:
                char_vec = self.char_cnn(char_ids)
                additional_vec = torch.cat([additional_vec, char_vec], dim=2)

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
                additional_vec = torch.cat([additional_vec, pattern_vec], dim=2)
            sequence_output = torch.cat([sequence_output, additional_vec], dim=2)

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
