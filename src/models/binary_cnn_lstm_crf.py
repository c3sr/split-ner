import argparse
import os

import numpy as np
import torch

from src.components.crf import CRF
from src.models.type_cnn_lstm_crf import TypeCNN_LSTM_CRFExecutor, TypeCRFDataset
from src.utils.evaluator import Evaluator
from src.utils.general import set_all_seeds


class BinaryCRFDataset(TypeCRFDataset):

    def __init__(self, corpus_path, word_vocab_path, out_tag_vocab_path, inp_tag_vocab_path, out_tag_names_path,
                 pos_tag_vocab_path, dep_tag_vocab_path, word_emb_path=None,
                 tag_emb_path=None, use_char="lower", use_pattern="condensed", use_word="glove",
                 include_word_lengths=False, retain_digits=False, pad_tag="<PAD>", none_tag="O", unk_tag="<UNK>",
                 info_tag="<INFO>", word_emb_dim=50, max_word_len=20, max_seq_len=20, post_padding=True,
                 use_tag_info="self", window_size=5):
        super(BinaryCRFDataset, self).__init__(corpus_path, word_vocab_path, out_tag_vocab_path, inp_tag_vocab_path,
                                               out_tag_names_path, pos_tag_vocab_path, dep_tag_vocab_path,
                                               word_emb_path, tag_emb_path, use_char, use_pattern, use_word,
                                               include_word_lengths, retain_digits, pad_tag, none_tag, unk_tag,
                                               word_emb_dim, max_word_len, max_seq_len, post_padding, use_tag_info,
                                               window_size)
        self.info_tag = info_tag
        self.model_training_out_tags = [self.pad_tag, self.none_tag, self.info_tag, CRF.START_TAG, CRF.STOP_TAG]

    def get_indexed_tag(self, text_tag):
        indexed_tag = []
        for curr_tag in text_tag:
            if curr_tag in self.model_training_out_tags:
                indexed_tag.append(self.model_training_out_tags.index(curr_tag))
            else:
                indexed_tag.append(self.model_training_out_tags.index(self.info_tag))
        return np.array(indexed_tag)


class BinaryCNN_LSTM_CRFExecutor(TypeCNN_LSTM_CRFExecutor):

    def define_datasets(self):
        post_padding = not self.args.use_pre_padding
        include_word_lengths = not self.args.ignore_word_lengths
        retain_digits = not self.args.escape_digits
        self.train_dataset = BinaryCRFDataset(corpus_path=self.args.train_path, out_tag_vocab_path=self.args.tags_path,
                                              word_vocab_path=self.args.word_vocab_path,
                                              out_tag_names_path=self.args.out_tag_names_path,
                                              pos_tag_vocab_path=self.args.pos_tag_vocab_path,
                                              dep_tag_vocab_path=self.args.dep_tag_vocab_path,
                                              word_emb_path=self.args.emb_path,
                                              tag_emb_path=self.args.tag_emb_path,
                                              unk_tag=self.unk_tag, pad_tag=self.pad_tag, none_tag=self.none_tag,
                                              use_char=self.args.use_char, use_word=self.args.use_word,
                                              use_pattern=self.args.use_pattern, word_emb_dim=self.args.word_emb_dim,
                                              max_word_len=self.args.max_word_len, max_seq_len=self.args.max_seq_len,
                                              post_padding=post_padding, retain_digits=retain_digits,
                                              include_word_lengths=include_word_lengths,
                                              use_tag_info=self.args.use_tag_info,
                                              inp_tag_vocab_path=self.args.inp_tag_vocab_path,
                                              window_size=self.args.window_size)
        # not parsing the embedding file again when processing the dev/test sets
        self.dev_dataset = BinaryCRFDataset(corpus_path=self.args.dev_path, out_tag_vocab_path=self.args.tags_path,
                                            word_vocab_path=self.args.word_vocab_path,
                                            out_tag_names_path=self.args.out_tag_names_path,
                                            pos_tag_vocab_path=self.args.pos_tag_vocab_path,
                                            dep_tag_vocab_path=self.args.dep_tag_vocab_path, word_emb_path=None,
                                            unk_tag=self.unk_tag,
                                            tag_emb_path=self.args.tag_emb_path,
                                            pad_tag=self.pad_tag, none_tag=self.none_tag, use_char=self.args.use_char,
                                            use_pattern=self.args.use_pattern, use_word=self.args.use_word,
                                            word_emb_dim=self.args.word_emb_dim, max_word_len=self.args.max_word_len,
                                            max_seq_len=self.args.max_seq_len, post_padding=post_padding,
                                            retain_digits=retain_digits,
                                            include_word_lengths=include_word_lengths,
                                            use_tag_info=self.args.use_tag_info,
                                            inp_tag_vocab_path=self.args.inp_tag_vocab_path,
                                            window_size=self.args.window_size)
        self.test_dataset = BinaryCRFDataset(corpus_path=self.args.test_path, out_tag_vocab_path=self.args.tags_path,
                                             word_vocab_path=self.args.word_vocab_path,
                                             out_tag_names_path=self.args.out_tag_names_path,
                                             pos_tag_vocab_path=self.args.pos_tag_vocab_path,
                                             dep_tag_vocab_path=self.args.dep_tag_vocab_path, word_emb_path=None,
                                             unk_tag=self.unk_tag,
                                             tag_emb_path=self.args.tag_emb_path,
                                             pad_tag=self.pad_tag, none_tag=self.none_tag, use_char=self.args.use_char,
                                             use_pattern=self.args.use_pattern, use_word=self.args.use_word,
                                             word_emb_dim=self.args.word_emb_dim, max_word_len=self.args.max_word_len,
                                             max_seq_len=self.args.max_seq_len, post_padding=post_padding,
                                             retain_digits=retain_digits,
                                             include_word_lengths=include_word_lengths,
                                             use_tag_info=self.args.use_tag_info,
                                             inp_tag_vocab_path=self.args.inp_tag_vocab_path,
                                             window_size=self.args.window_size)

    def get_model_training_out_dim(self):
        return len(self.train_dataset.model_training_out_tags)

    def get_model_training_out_tags(self):
        return self.train_dataset.model_training_out_tags

    def run_pipeline(self, pipeline_dir):
        self.resume_checkpoint()
        self.print_model_summary()

        pipeline_dir = os.path.join(self.args.data_dir, pipeline_dir, self.args.name)
        if not os.path.exists(pipeline_dir):
            os.makedirs(pipeline_dir)

        train_outfile = os.path.join(pipeline_dir, "train.out.tsv")
        dev_outfile = os.path.join(pipeline_dir, "dev.out.tsv")
        test_outfile = os.path.join(pipeline_dir, "test.out.tsv")
        train_evaluator = self.generate_outputs_for_pipeline(self.train_data_loader, outfile=train_outfile)
        dev_evaluator = self.generate_outputs_for_pipeline(self.dev_data_loader, outfile=dev_outfile)
        test_evaluator = self.generate_outputs_for_pipeline(self.test_data_loader, outfile=test_outfile)

        return train_evaluator, dev_evaluator, test_evaluator

    def generate_outputs_for_pipeline(self, data_loader, outfile):
        # TODO: assert that shuffle is false
        self.model.eval()
        total_text = []
        total_prediction = []
        total_label = []
        for text, word_feature, char_feature, type_feature, word_mask, char_mask, label in data_loader:
            text = np.array(text).T.tolist()
            word_feature = word_feature.to(self.device)
            char_feature = char_feature.to(self.device)
            type_feature = type_feature.to(self.device)
            word_mask = word_mask.to(self.device)
            char_mask = char_mask.to(self.device)
            label = label.to(self.device)
            with torch.no_grad():
                _, prediction = self.model(text, word_feature, char_feature, type_feature, word_mask, char_mask)
                total_text.extend(text)
                total_prediction.append(prediction)
                total_label.append(label.clone())
        total_prediction = np.vstack([np.array(p) for p in total_prediction])
        total_label = torch.cat(total_label, dim=0).cpu().numpy()
        self.print_outputs_for_pipeline(corpus=total_text, predicted=total_prediction, dataset=data_loader.dataset,
                                        outfile=outfile)
        evaluator = Evaluator(gold=total_label, predicted=total_prediction, tags=self.get_model_training_out_tags(),
                              ignore_tags=[self.none_tag, self.pad_tag], none_tag=self.none_tag, pad_tag=self.pad_tag)
        if self.args.verbose:
            print("Entity-Level Metrics:")
            print(evaluator.entity_metric.report())
            print("Token-Level Metrics:")
            print(evaluator.significant_token_metric.report())

        return evaluator

    def print_outputs_for_pipeline(self, corpus, predicted, dataset, outfile):
        print("Outputs published in file: {0}".format(outfile))
        with open(outfile, "w") as f:
            for sent_index in range(len(corpus)):
                for word_index in range(len(corpus[sent_index])):
                    token = corpus[sent_index][word_index]
                    fine_grained_gold_tag = dataset.text_tags[sent_index][word_index]
                    predicted_tag = dataset.model_training_out_tags[predicted[sent_index][word_index]]
                    if predicted_tag == dataset.info_tag:
                        if fine_grained_gold_tag == self.none_tag:
                            # any class tag. It is anyways wrong. So, the error is propagating
                            output_tag = dataset.out_tags[2]
                        else:
                            output_tag = fine_grained_gold_tag
                    else:
                        output_tag = predicted_tag
                    if output_tag != self.pad_tag:
                        f.write("{0}\t{1}\n".format(token, output_tag))
                f.write("\n")


def main(args):
    set_all_seeds(args.seed)
    executor = BinaryCNN_LSTM_CRFExecutor(args)
    executor.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Binary CNN-LSTM-CRF Model for Sequence Labeling")
    ap.add_argument("--name", type=str, default="binary-cnn-lstm-crf",
                    help="model name (Default: 'binary-cnn-lstm-crf')")
    ap.add_argument("--checkpoint_dir", type=str, default="../../checkpoints",
                    help="checkpoints directory (Default: '../../checkpoints')")
    ap.add_argument("--eval", type=str, default="none",
                    help="only evaluate existing checkpoint model (none/best/<checkpoint-id>) (Default: 'none')")
    ap.add_argument("--query", action="store_true",
                    help="query mode, can be used with eval to work with best model (Default: False)")

    ap.add_argument("--data_dir", type=str, default="../../data/GENIA_term_3.02",
                    help="path to input dataset directory (Default: '../../data/GENIA_term_3.02')")
    ap.add_argument("--out_dir", type=str, default="../../data/GENIA_term_3.02/out",
                    help="path to output directory (Default: '../../data/GENIA_term_3.02/out')")
    ap.add_argument("--train_path", type=str, default="train.tsv",
                    help="path to train dataset (train.tsv|std_train.tsv) (Default: 'train.tsv')")
    ap.add_argument("--dev_path", type=str, default="dev.tsv",
                    help="path to dev dataset (dev.tsv|std_dev.tsv) (Default: 'dev.tsv')")
    ap.add_argument("--test_path", type=str, default="test.tsv",
                    help="path to test dataset (test.tsv|std_test.tsv) (Default: 'test.tsv')")
    ap.add_argument("--word_vocab_path", type=str, default="glove_vocab.txt",
                    help="path to word vocab (Default: 'glove_vocab.txt')")
    ap.add_argument("--tags_path", type=str, default="tag_vocab.txt",
                    help="path to output tags vocab. Use 'tag_vocab.txt' for full tags vocab. "
                         "Use 'std_tag_vocab.txt' for standard 5 tags vocab. "
                         "Use 'out_freq_tag_vocab.txt' for reduced tags, when considering input tags information. "
                         "(Default: 'tag_vocab.txt')")
    ap.add_argument("--out_tag_names_path", type=str, default="tag_names.txt",
                    help="path to output tag general names. Use 'tag_names.txt' for full tags vocab names. "
                         "Use 'std_tag_names.txt' for standard 5 tags vocab names. "
                         "Use 'jnlpba_tag_names.txt' for exact (5-tag) settings used by MTL-BioInformatics-2016 "
                         "(ref: https://github.com/cambridgeltl/MTL-Bioinformatics-2016)"
                         "Use 'out_freq_tag_names.txt' for reduced tags, when considering input tags information. "
                         "(Default: 'tag_names.txt')")
    ap.add_argument("--inp_tag_vocab_path", type=str, default="empty_inp_tag_vocab.txt",
                    help="path to input tags vocab. Use 'empty_inp_tag_vocab.txt' if don't want to use tag info. "
                         "Use 'inp_freq_tag_vocab.txt' for specifying default input tag info."
                         "(Default: 'empty_inp_tag_vocab.txt')")
    ap.add_argument("--pos_tag_vocab_path", type=str, default="pos_tag_vocab.txt",
                    help="path to POS tags vocab. (pos_tag_vocab.txt|jnlpba_pos_tag_vocab.txt) "
                         "(Default: 'pos_tag_vocab.txt')")
    ap.add_argument("--dep_tag_vocab_path", type=str, default="dep_tag_vocab.txt",
                    help="path to dependency-parse tags vocab. (dep_tag_vocab.txt|jnlpba_dep_tag_vocab.txt) "
                         "(Default: 'dep_tag_vocab.txt')")
    ap.add_argument("--emb_path", type=str, default="../../../../Embeddings/glove.6B.50d.txt",
                    help="path to pre-trained word embeddings (Default: '../../../../Embeddings/glove.6B.50d.txt')")
    ap.add_argument("--tag_emb_path", type=str, default="tag_w2v_emb.txt",
                    help="path to pre-trained tag embeddings, relative to data_dir "
                         "(std_tag_w2v_emb.txt|std_tag_use_emb.txt|std_tag_full_emb.txt) "
                         "(tag_w2v_emb.txt|tag_use_emb.txt|tag_full_emb.txt) (Default: 'tag_w2v_emb.txt')")

    ap.add_argument("--num_epochs", type=int, default=500, help="# epochs to train (Default: 500)")
    ap.add_argument("--batch_size", type=int, default=128, help="batch size (Default: 128)")
    ap.add_argument("--word_emb_dim", type=int, default=50, help="word embedding dimension (Default: 50)")
    ap.add_argument("--pos_tag_emb_dim", type=int, default=15, help="POS tag embedding dimension (Default: 15)")
    ap.add_argument("--dep_tag_emb_dim", type=int, default=15, help="dep-parse tag embedding dimension (Default: 15)")
    ap.add_argument("--max_word_len", type=int, default=30, help="max. #chars in word (Default: 30)")
    ap.add_argument("--max_seq_len", type=int, default=60, help="max. #words in sentence (Default: 60)")
    ap.add_argument("--conv1_dim", type=int, default=128, help="conv1 layer output channels (Default: 128)")
    ap.add_argument("--hidden_dim", type=int, default=256, help="hidden state dim for LSTM, if used (Default: 256)")
    ap.add_argument("--use_pos_tag", action="store_true", help="embed POS tag information (Default: False)")
    ap.add_argument("--use_dep_tag", action="store_true", help="embed dep-parse tag information (Default: False)")
    ap.add_argument("--use_maxpool", action="store_true", help="max pool over CNN output to get char embeddings, else "
                                                               "does concatenation (Default: False)")
    ap.add_argument("--use_tag_cosine_sim", action="store_true",
                    help="compute cosine sim with tag embeddings as additional layer in model (Default: False)")
    ap.add_argument("--kernel_size", type=int, default=5, help="kernel size for CNN (Default: 5)")
    ap.add_argument("--num_lstm_layers", type=int, default=1, help="no. of LSTM layers (Default: 1)")
    ap.add_argument("--use_char", type=str, default="lower",
                    help="char embedding type (none/lower/all) (Default: 'lower')")
    ap.add_argument("--use_pattern", type=str, default="condensed",
                    help="pattern embedding type (none/one-to-one/condensed) (Default: 'condensed')")
    ap.add_argument("--escape_digits", action="store_true",
                    help="replace digits(0-9) with 'd' tag in pattern capturing (Default: False)")
    ap.add_argument("--ignore_word_lengths", action="store_true",
                    help="ignore word lengths in pattern capturing (Default: False)")
    ap.add_argument("--no_lstm", action="store_true",
                    help="don't use LSTM to capture neighbor context. Directly CRF over individual token level CNN "
                         "(Default: False)")
    ap.add_argument("--use_tag_info", type=str, default="none",
                    help="type information (none/self/window) (Default: 'none')")
    ap.add_argument("--use_tfo", type=str, default="none",
                    help="use transformer (may not use LSTM then). 'simple' creates a basic tfo. "
                         "'xl' uses TransformerXL model layer. (none|simple|xl) (Default: 'none')")
    ap.add_argument("--window_size", type=int, default=5,
                    help="size of context window for type info on either side of current token (Default: 5)")
    ap.add_argument("--use_word", type=str, default="allenai/scibert_scivocab_uncased",
                    help="use word(token) embeddings "
                         "(none|rand|glove|allenai/scibert_scivocab_uncased|bert-base-uncased"
                         "|../../../resources/biobert_v1.1_pubmed) "
                         "(Default: allenai/scibert_scivocab_uncased)")
    ap.add_argument("--use_pre_padding", action="store_true", help="pre-padding for char/word (Default: False)")
    ap.add_argument("--word_emb_model_from_tf", action="store_true",
                    help="word embedding generator model is a pretrained tensorflow model. Use 'True' for models like, "
                         "'../../../resources/biobert_v1.1_pubmed' (Default: False)")
    ap.add_argument("--use_class_guidance", action="store_true",
                    help="take guidance through pre-trained class embeddings (Default: False)")
    ap.add_argument("--fine_tune_bert", action="store_true", help="fine-tune bert embeddings (Default: False)")
    ap.add_argument("--lr", type=float, default=0.001, help="learning rate (Default: 0.001)")
    ap.add_argument("--seed", type=int, default=42, help="manual seed for reproducibility (Default: 42)")
    ap.add_argument("--use_cpu", action="store_true", help="force CPU usage (Default: False)")
    ap.add_argument("--no_eval_print", action="store_true",
                    help="don't output verbose evaluation matrices (Default: False)")
    ap = ap.parse_args()
    main(ap)
