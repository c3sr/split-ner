import torch
from torch import nn as nn
from torch.nn import functional as F
from transformers import BertTokenizer, BertModel, TransfoXLModel, TransfoXLConfig


class CNN_LSTM_Base(nn.Module):
    EXPAND_FACTOR = 5

    def __init__(self, inp_dim, conv1_dim, hidden_dim, kernel_size, word_len, device, word_vocab_size=None,
                 pos_tag_vocab_size=None, dep_tag_vocab_size=None, word_emb_dim=None, pos_tag_emb_dim=None,
                 dep_tag_emb_dim=None, tag_emb_dim=None, pre_trained_emb=None, use_word="glove", use_pos_tag=False,
                 use_dep_tag=False, use_char=True, use_maxpool=False, use_lstm=True, use_tag_info="self",
                 use_tag_cosine_sim=False, dropout_ratio=0.5, fine_tune_bert=False, use_tfo="none", tag_emb=None,
                 word_emb_model_from_tf=False, num_lstm_layers=1):
        super(CNN_LSTM_Base, self).__init__()

        self.use_maxpool = use_maxpool
        self.use_word = use_word
        self.use_char = use_char
        self.use_lstm = use_lstm
        self.use_tag_info = use_tag_info
        self.hidden_dim = hidden_dim
        self.device = device
        self.use_tag_cosine_sim = use_tag_cosine_sim
        self.use_pos_tag = use_pos_tag
        self.use_dep_tag = use_dep_tag
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fine_tune_bert = fine_tune_bert
        self.use_tfo = use_tfo
        self.tag_emb = tag_emb

        assert self.use_char or self.use_word != "none", "either of char or word embeddings need to be used"

        if self.use_char:
            print("model using char embeddings")
        if self.use_word != "none":
            print("model using word embeddings")

        next_inp_dim = 0

        if self.use_char:
            self.conv1 = nn.Conv1d(in_channels=inp_dim, out_channels=conv1_dim, kernel_size=kernel_size)

            if self.use_maxpool:
                next_inp_dim += conv1_dim
            else:
                next_inp_dim += (word_len - (kernel_size - 1) * 1) * conv1_dim

        if self.use_word != "none":
            if self.use_word == "rand":
                assert word_vocab_size, "word vocab size needs to be specified"
                assert word_emb_dim, "word embedding dimension needs to be specified"
                self.emb = nn.Embedding(word_vocab_size, word_emb_dim)

            elif self.use_word == "glove":
                assert isinstance(pre_trained_emb, torch.Tensor), "pre-trained glove embeddings need to be specified"
                self.emb = nn.Embedding.from_pretrained(embeddings=pre_trained_emb, freeze=False)
                word_emb_dim = pre_trained_emb.shape[1]

            elif "bert" in self.use_word:
                # possible values include: "bert-base-uncased", "allenai/scibert_scivocab_uncased"
                self.bert_tokenizer = BertTokenizer.from_pretrained(self.use_word)
                self.bert_model = BertModel.from_pretrained(self.use_word, from_tf=word_emb_model_from_tf)
                word_emb_dim = self.bert_model.config.hidden_size
                if self.use_tfo == "xl":
                    self.tfo_model = TransfoXLModel(TransfoXLConfig(n_layer=1, d_model=word_emb_dim, d_inner=256))
                else:
                    self.tfo_model = None
                if not self.fine_tune_bert:
                    for param in self.bert_model.parameters():
                        param.requires_grad = False

            next_inp_dim += word_emb_dim

        if self.use_tag_info != "none":
            assert tag_emb_dim, "tag vocab size needs to be specified"
            next_inp_dim += tag_emb_dim

        if self.use_pos_tag:
            assert pos_tag_vocab_size, "POS tag vocab size needs to be specified"
            assert pos_tag_emb_dim, "POS tag embedding dimension needs to be specified"
            self.pos_tag_emb = nn.Embedding(pos_tag_vocab_size, pos_tag_emb_dim)
            next_inp_dim += pos_tag_emb_dim

        if self.use_dep_tag:
            assert dep_tag_vocab_size, "dep-parse tag vocab size needs to be specified"
            assert dep_tag_emb_dim, "dep-parse tag embedding dimension needs to be specified"
            self.dep_tag_emb = nn.Embedding(dep_tag_vocab_size, dep_tag_emb_dim)
            next_inp_dim += dep_tag_emb_dim

        # TODO: can add a Conv layer (over words) here (instead of LSTM)
        if self.use_lstm:
            self.lstm = nn.LSTM(input_size=next_inp_dim, hidden_size=self.hidden_dim, bidirectional=True,
                                batch_first=True, num_layers=num_lstm_layers, dropout=dropout_ratio)
            next_inp_dim = self.hidden_dim * 2

        if self.use_tfo == "simple":
            tfo_layer = nn.TransformerEncoderLayer(d_model=next_inp_dim, nhead=4, dim_feedforward=256)
            self.tfo = nn.TransformerEncoder(encoder_layer=tfo_layer, num_layers=2)

        if use_tag_cosine_sim:
            assert isinstance(self.tag_emb,
                              torch.Tensor), "tag embeddings tensor is needed for tag cosine similarity calculations"
            self.num_tags, self.tag_emb_dim = self.tag_emb.shape
            self.fc2 = nn.Linear(in_features=next_inp_dim, out_features=self.tag_emb_dim)
            self.cos = nn.CosineSimilarity(dim=3)
            next_inp_dim += self.num_tags

        self.next_inp_dim = next_inp_dim

    # TODO: Check. This is not being used right now. Do we need to use it in forward?
    def init_hidden(self, batch_size):
        return torch.randn(2, batch_size, self.hidden_dim, device=self.device), \
               torch.randn(2, batch_size, self.hidden_dim, device=self.device)

    def forward(self, text, word_x, char_x, type_x, word_mask, char_mask):
        batch_size, seq_len, word_len, inp_emb_dim = char_x.shape

        x = torch.Tensor().to(word_x.device)
        if self.use_char:
            char_x = char_x.view(batch_size * seq_len, word_len, inp_emb_dim)
            char_x = char_x.transpose(1, 2)
            char_x = self.conv1(char_x)
            char_x = F.relu(char_x)

            # Note: Removing ReLU leads to slight reduction in performance

            if self.use_maxpool:
                char_x, _ = char_x.max(dim=2)
            char_x = char_x.view(batch_size, seq_len, -1)
            x = torch.cat((x, char_x), dim=2)

        if self.use_word != "none":
            if "bert" in self.use_word:
                word_emb_x = CNN_LSTM_Base.get_bert_embeddings(text, seq_len, self.bert_tokenizer, self.bert_model,
                                                               self.device, self.use_tfo, self.tfo_model)
            else:
                word_emb_x = self.emb(word_x[:, :, 0])
            x = torch.cat((x, word_emb_x), dim=2)

        if self.use_tag_info != "none":
            x = torch.cat((x, type_x), dim=2)

        if self.use_pos_tag:
            pos_tag_x = self.pos_tag_emb(word_x[:, :, 1].to(torch.int64))
            x = torch.cat((x, pos_tag_x), dim=2)

        if self.use_dep_tag:
            dep_tag_x = self.dep_tag_emb(word_x[:, :, 2].to(torch.int64))
            x = torch.cat((x, dep_tag_x), dim=2)

        x = self.dropout(x)

        apply_dropout = False
        if self.use_lstm:
            apply_dropout = True
            lengths = torch.as_tensor(word_mask.sum(1).int(), dtype=torch.int64, device=torch.device("cpu"))
            packed_inp = nn.utils.rnn.pack_padded_sequence(input=x, lengths=lengths,
                                                           batch_first=True,
                                                           enforce_sorted=False)
            packed_out, _ = self.lstm(packed_inp)
            x, _ = nn.utils.rnn.pad_packed_sequence(sequence=packed_out, batch_first=True, total_length=seq_len)

        if self.use_tfo == "simple":
            apply_dropout = True
            x = self.tfo(x)

        if self.use_tag_cosine_sim:
            apply_dropout = True
            mod_x = self.fc2(x)
            mod_x = mod_x.unsqueeze(2).expand(batch_size, seq_len, self.num_tags, self.tag_emb_dim)
            t = self.tag_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, self.num_tags, self.tag_emb_dim)
            sim_x = self.cos(mod_x, t)
            x = torch.cat((x, sim_x), dim=2)

        if apply_dropout:
            x = self.dropout(x)

        return x

    @staticmethod
    def get_bert_embeddings(batch_text, seq_len, bert_tokenizer, bert_model, device, use_tfo="none", tfo_model=None):
        bert_batch_tokens = bert_tokenizer(batch_text, is_pretokenized=True, return_tensors="pt", padding="max_length",
                                           max_length=(CNN_LSTM_Base.EXPAND_FACTOR * seq_len))["input_ids"]
        bert_batch_vectors = bert_model(bert_batch_tokens.to(device))[0]

        if use_tfo == "xl":
            bert_batch_vectors = tfo_model(inputs_embeds=bert_batch_vectors)[0]

        emb = []
        for sent_index in range(len(batch_text)):
            sent = [[token] for token in batch_text[sent_index]]
            bert_sent_tokens = bert_tokenizer(sent, is_pretokenized=True)["input_ids"]
            word_ending = [max(1, len(bert_word_tokens) - 2) for bert_word_tokens in bert_sent_tokens]
            word_ending[0] -= 1
            for i in range(1, len(word_ending)):
                word_ending[i] += word_ending[i - 1]
            sent_emb = []
            bert_sent_vectors = bert_batch_vectors[sent_index][1:]
            curr_word_index = 0
            curr_word_vectors = []
            for token_index in range(len(bert_sent_vectors)):
                bert_token_vector = bert_sent_vectors[token_index].to(device)
                curr_word_vectors.append(bert_token_vector)
                if word_ending[curr_word_index] == token_index:
                    sent_emb.append(torch.mean(torch.stack(curr_word_vectors), dim=0))
                    curr_word_vectors = []
                    curr_word_index += 1
                    if curr_word_index == len(word_ending):
                        break
            emb.append(torch.stack(sent_emb))
        emb = torch.stack(emb)
        return emb


class CNN_LSTM(nn.Module):

    def __init__(self, inp_dim, conv1_dim, hidden_dim, out_dim, kernel_size, word_len, device, word_vocab_size=None,
                 pos_tag_vocab_size=None, dep_tag_vocab_size=None, word_emb_dim=None, pos_tag_emb_dim=None,
                 dep_tag_emb_dim=None, tag_emb_dim=None, pre_trained_emb=None, use_word="glove", use_pos_tag=False,
                 use_dep_tag=False, use_char=True, use_maxpool=False, use_lstm=True, use_tag_info="self",
                 use_tag_cosine_sim=False, dropout_ratio=0.5, fine_tune_bert=False, use_tfo="none",
                 use_class_guidance=False, tag_emb=None, word_emb_model_from_tf=False, num_lstm_layers=1):
        super(CNN_LSTM, self).__init__()

        self.base = CNN_LSTM_Base(inp_dim, conv1_dim, hidden_dim, kernel_size, word_len, device,
                                  word_vocab_size, pos_tag_vocab_size, dep_tag_vocab_size, word_emb_dim,
                                  pos_tag_emb_dim, dep_tag_emb_dim, tag_emb_dim, pre_trained_emb, use_word, use_pos_tag,
                                  use_dep_tag, use_char, use_maxpool, use_lstm, use_tag_info, use_tag_cosine_sim,
                                  dropout_ratio, fine_tune_bert, use_tfo, tag_emb, word_emb_model_from_tf,
                                  num_lstm_layers)
        self.use_class_guidance = use_class_guidance
        if self.use_class_guidance:
            assert isinstance(tag_emb, torch.Tensor), "tag embeddings tensor is needed for class guidance calculations"
            self.fc1 = ClassGuidedClassifier(inp_dim=self.base.next_inp_dim, out_dim=out_dim, hidden_dim=96,
                                             class_guidance=tag_emb,
                                             dropout_ratio=dropout_ratio)
        else:
            self.fc1 = nn.Linear(in_features=self.base.next_inp_dim, out_features=out_dim)

    def forward(self, text, word_x, char_x, type_x, word_mask, char_mask):
        x = self.base(text, word_x, char_x, type_x, word_mask, char_mask)
        x = self.fc1(x)
        x = x * word_mask.unsqueeze(-1).expand_as(x)

        return x


class CNN_LSTM_Span(nn.Module):

    def __init__(self, inp_dim, conv1_dim, hidden_dim, out_dim, kernel_size, word_len, device, word_vocab_size=None,
                 pos_tag_vocab_size=None, dep_tag_vocab_size=None, word_emb_dim=None, pos_tag_emb_dim=None,
                 dep_tag_emb_dim=None, tag_emb_dim=None, pre_trained_emb=None, use_word="glove", use_pos_tag=False,
                 use_dep_tag=False, use_char=True, use_maxpool=False, use_lstm=True, use_tag_info="self",
                 use_tag_cosine_sim=False, dropout_ratio=0.5, fine_tune_bert=False, use_tfo="none",
                 use_class_guidance=False, tag_emb=None, span_pooling="boundary", word_emb_model_from_tf=False,
                 num_lstm_layers=1):
        super(CNN_LSTM_Span, self).__init__()

        self.base = CNN_LSTM_Base(inp_dim, conv1_dim, hidden_dim, kernel_size, word_len, device,
                                  word_vocab_size, pos_tag_vocab_size, dep_tag_vocab_size, word_emb_dim,
                                  pos_tag_emb_dim, dep_tag_emb_dim, tag_emb_dim, pre_trained_emb, use_word, use_pos_tag,
                                  use_dep_tag, use_char, use_maxpool, use_lstm, use_tag_info, use_tag_cosine_sim,
                                  dropout_ratio, fine_tune_bert, use_tfo, tag_emb, word_emb_model_from_tf,
                                  num_lstm_layers)

        self.use_class_guidance = use_class_guidance
        self.begin_outputs = nn.Linear(self.base.next_inp_dim, 2)
        self.end_outputs = nn.Linear(self.base.next_inp_dim, 2)
        self.span_pooling = span_pooling
        self.device = device

        if self.span_pooling == "avg":
            self.span_dim = self.base.next_inp_dim
        else:  # self.span_pooling == "boundary":
            self.span_dim = self.base.next_inp_dim * 2

        if self.use_class_guidance:
            assert isinstance(tag_emb, torch.Tensor), "tag embeddings tensor is needed for class guidance calculations"
            self.span_embedding = ClassGuidedClassifier(inp_dim=self.span_dim, out_dim=out_dim, hidden_dim=96,
                                                        class_guidance=tag_emb, dropout_ratio=dropout_ratio)
        else:
            self.span_embedding = MultiClassSpanClassifier(self.span_dim, out_dim, dropout_ratio)

    def forward(self, text, word_x, char_x, type_x, word_mask, char_mask):
        batch_size, seq_len, word_len, inp_emb_dim = char_x.shape

        x = self.base(text, word_x, char_x, type_x, word_mask, char_mask)  # B x S x next_inp_dim
        begin_x = self.begin_outputs(x).squeeze()
        end_x = self.end_outputs(x).squeeze()

        if self.span_pooling == "avg":
            cumsum_x = torch.cumsum(x, dim=1)
            span_matrix = torch.zeros((batch_size, seq_len, seq_len, self.span_dim), dtype=torch.float32,
                                      device=self.device)
            for start in range(seq_len):
                for end in range(start, seq_len):
                    span_matrix[:, start, end, :] = cumsum_x[:, end, :]
                    if start >= 1:
                        span_matrix[:, start, end, :] -= cumsum_x[:, start - 1, :]
                    span_matrix[:, start, end, :] /= (end - start + 1)
        else:  # self.span_pooling == "boundary":
            begin_extend = x.unsqueeze(2).expand(-1, -1, seq_len, -1)
            end_extend = x.unsqueeze(1).expand(-1, seq_len, -1, -1)
            span_matrix = torch.cat([begin_extend, end_extend], 3)  # B x S x S x 2*next_inp_dim

        span_x = self.span_embedding(span_matrix)  # B x S x S x T

        return begin_x, end_x, span_x


class ClassGuidedClassifier(nn.Module):

    def __init__(self, inp_dim, hidden_dim, out_dim, class_guidance, dropout_ratio):
        super(ClassGuidedClassifier, self).__init__()
        self.class_guidance = class_guidance
        self.num_classes, self.class_emb_dim = self.class_guidance.shape
        self.fc1 = nn.Linear(in_features=inp_dim + self.class_emb_dim, out_features=hidden_dim)
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc2 = nn.Linear(in_features=self.num_classes * hidden_dim, out_features=out_dim)

    def forward(self, x):
        num_dims = len(x.shape) + 1
        expanded_shape = [-1] * num_dims
        expanded_shape[-2] = self.num_classes
        x = x.unsqueeze(-2).expand(*expanded_shape)
        class_x = self.class_guidance
        base_shape = []
        for i in range(num_dims - 2):
            class_x = class_x.unsqueeze(0)
            base_shape.append(x.shape[i])
        class_x = class_x.expand(*(base_shape + [-1, -1]))
        x = torch.cat((x, class_x), dim=-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = x.reshape(*(base_shape + [-1]))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MultiClassSpanClassifier(nn.Module):

    def __init__(self, inp_dim, out_dim, dropout_ratio):
        super(MultiClassSpanClassifier, self).__init__()
        self.fc1 = nn.Linear(inp_dim, inp_dim // 2)
        self.fc2 = nn.Linear(inp_dim // 2, out_dim)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
