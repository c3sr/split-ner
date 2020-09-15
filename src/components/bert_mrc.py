import torch
from torch import nn as nn
from torch.nn import functional as F
from transformers import BertTokenizer, BertModel

from src.components.cnn_lstm import CNN_LSTM_Base


class BERT_MRC(nn.Module):

    def __init__(self, device, out_tag_names, hidden_dim, out_dim, use_word="allenai/scibert_scivocab_uncased",
                 word_emb_model_from_tf=False, dropout_ratio=0.5):
        super(BERT_MRC, self).__init__()

        self.device = device
        self.out_tag_names = out_tag_names
        self.use_word = use_word
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.word_emb_model_from_tf = word_emb_model_from_tf

        # possible values of self.use_word include: "bert-base-uncased", "allenai/scibert_scivocab_uncased"
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.use_word)
        self.bert_model = BertModel.from_pretrained(self.use_word, from_tf=word_emb_model_from_tf)
        for param in self.bert_model.parameters():
            param.requires_grad = False

        next_inp_dim = self.bert_model.config.hidden_size

        self.num_tokens_in_out_tag_name = self.get_num_bert_tokens_in_out_tag_name()

        self.fc1 = nn.Linear(in_features=next_inp_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=len(self.out_tag_names) * hidden_dim, out_features=out_dim)

    def forward(self, text, word_x, char_x, type_x, word_mask, char_mask):
        batch_size, seq_len, word_len, inp_emb_dim = char_x.shape

        x_list = []
        for i, out_tag_name in enumerate(self.out_tag_names):
            new_text = [[out_tag_name] + sent for sent in text]
            x_list.append(CNN_LSTM_Base.get_bert_embeddings(new_text, seq_len + 1,
                                                            self.bert_tokenizer, self.bert_model, self.device)[:, 1:,
                          :])
        x = torch.stack(x_list, dim=2)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = x.reshape(batch_size, seq_len, -1)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_num_bert_tokens_in_out_tag_name(self):
        return [len(self.bert_tokenizer(out_tag_name)["input_ids"][1:-1]) for out_tag_name in self.out_tag_names]
