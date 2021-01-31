import torch
import torch.nn as nn
import torch.nn.functional as F
from secner.additional_args import AdditionalArguments
from secner.dataset import NerDataset
from transformers import BertConfig
from transformers.models.bert import BertPreTrainedModel


class NerModelChar(BertPreTrainedModel):

    def __init__(self, config: BertConfig, additional_args: AdditionalArguments):
        super(NerModelChar, self).__init__(config)
        self.additional_args = additional_args
        self.num_labels = config.num_labels

        # Character Embedding Layer
        num_embeddings = len(NerDataset.get_char_vocab()) + 1  # +1 (for the special [PAD] char)
        self.char_emb = nn.Embedding(num_embeddings=num_embeddings,
                                     embedding_dim=self.additional_args.char_emb_dim,
                                     padding_idx=0)
        # nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)

        self.char_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=self.additional_args.cnn_num_filters // 2,
                      kernel_size=(self.additional_args.char_emb_dim, self.additional_args.cnn_kernel_size)),
            nn.ReLU()
        )

        self.char_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=self.additional_args.cnn_num_filters,
                      kernel_size=(self.additional_args.cnn_num_filters // 2, self.additional_args.cnn_kernel_size)),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.hidden_size = self.additional_args.cnn_num_filters
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            char_ids=None,
            labels=None,
            **kwargs):

        sequence_output = self.char_emb_layer(input_ids)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (torch.argmax(logits, dim=2),)
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
        x = self.char_conv1(x)
        x = x.permute(0, 2, 1, 3)
        x = self.char_conv2(x)
        x = x.squeeze()
        # (batch * seq_len, char_channel_size, 1) -> (batch * seq_len, char_channel_size)
        x = F.max_pool1d(x, x.size(2)).squeeze()
        # (batch, seq_len, char_channel_size)
        x = x.view(batch_size, -1, self.additional_args.cnn_num_filters)

        return x
