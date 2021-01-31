import torch
import torch.nn as nn
import torch.nn.functional as F
from secner.additional_args import AdditionalArguments
from secner.dataset import NerDataset
from transformers import BertConfig
from transformers.models.bert import BertModel, BertPreTrainedModel


class NerModelBiDAF(BertPreTrainedModel):

    def __init__(self, config: BertConfig, additional_args: AdditionalArguments):
        super(NerModelBiDAF, self).__init__(config)
        self.additional_args = additional_args
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

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

        # highway network
        self.hidden_size = self.additional_args.cnn_num_filters + config.hidden_size
        for i in range(2):
            setattr(self, 'highway_linear{}'.format(i),
                    nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                  self.dropout,
                                  nn.ReLU()))
            setattr(self, 'highway_gate{}'.format(i),
                    nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                  self.dropout,
                                  nn.Sigmoid()))

        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

        self.init_weights()

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
            labels=None,
            **kwargs):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]

        char_vec = self.char_emb_layer(char_ids)
        sequence_output = self.highway_network(sequence_output, char_vec)

        logits = self.classifier(sequence_output)

        outputs = (torch.argmax(logits, dim=2),) + outputs[2:]  # add hidden states and attention if they are here
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
