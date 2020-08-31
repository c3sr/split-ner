import argparse
import copy
import os

import numpy as np

from src.models.binary_cnn_lstm_crf import BinaryCNN_LSTM_CRFExecutor
from src.models.cnn_cosine import TypeCNN_LSTMCustomLossExecutor
from src.utils.evaluator import Evaluator
from src.utils.general import set_all_seeds


def main(args):
    assert args.eval != "none", "use this pipeline only for evaluation. Train models independently."

    set_all_seeds(args.seed)

    args1 = copy.deepcopy(args)
    args2 = copy.deepcopy(args)

    # set specific args from parent args
    args1.name = args.name1
    args1.tags_path = args.tags_path1
    args1.use_maxpool = args.use_maxpool1
    args1.use_tag_cosine_sim = args.use_tag_cosine_sim1

    args2.name = args.name2
    args2.tags_path = args.tags_path2
    args2.use_maxpool = args.use_maxpool2
    args2.use_tag_cosine_sim = args.use_tag_cosine_sim2

    # connect the pipeline
    args2.train_path = os.path.join(args.pipeline_dir, args1.name, "train.out.tsv")
    args2.dev_path = os.path.join(args.pipeline_dir, args1.name, "dev.out.tsv")
    args2.test_path = os.path.join(args.pipeline_dir, args1.name, "test.out.tsv")

    # each intermediate step of pipeline can 'run_pipeline' and output results for input by next module
    executor1 = BinaryCNN_LSTM_CRFExecutor(args1)
    executor1.run_pipeline(args1.pipeline_dir)

    # final module just calls 'run' to get results in evaluator response form
    executor2 = TypeCNN_LSTMCustomLossExecutor(args2)
    train_evaluator2, dev_evaluator2, test_evaluator2 = executor2.run()

    # get input gold tags
    train_gold_text_tags = executor1.train_dataset.text_tags
    dev_gold_text_tags = executor1.dev_dataset.text_tags
    test_gold_text_tags = executor1.test_dataset.text_tags

    # get final predicted tags
    train_predicted_text_tags = generate_text_tags(executor2.train_dataset, train_evaluator2)
    dev_predicted_text_tags = generate_text_tags(executor2.dev_dataset, dev_evaluator2)
    test_predicted_text_tags = generate_text_tags(executor2.test_dataset, test_evaluator2)

    # map to tag indices (for evaluation)
    out_tags = executor1.train_dataset.out_tags
    none_tag = executor1.none_tag
    pad_tag = executor1.pad_tag

    train_gold = convert_to_indexed_tags(train_gold_text_tags, out_tags)
    train_predicted = convert_to_indexed_tags(train_predicted_text_tags, out_tags)

    dev_gold = convert_to_indexed_tags(dev_gold_text_tags, out_tags)
    dev_predicted = convert_to_indexed_tags(dev_predicted_text_tags, out_tags)

    test_gold = convert_to_indexed_tags(test_gold_text_tags, out_tags)
    test_predicted = convert_to_indexed_tags(test_predicted_text_tags, out_tags)

    # evaluate complete pipeline
    results_dir = os.path.join(args.out_dir, "pipeline_{0}_{1}".format(args.name1, args.name2))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    train_text_sentences = executor1.train_dataset.text_sentences
    train_outfile = os.path.join(results_dir, "train.out.tsv")
    evaluate_pipeline(executor1, train_gold, train_predicted, train_text_sentences, train_outfile, "TRAIN", out_tags,
                      none_tag,
                      pad_tag)

    dev_text_sentences = executor1.dev_dataset.text_sentences
    dev_outfile = os.path.join(results_dir, "dev.out.tsv")
    evaluate_pipeline(executor1, dev_gold, dev_predicted, dev_text_sentences, dev_outfile, "DEV", out_tags, none_tag,
                      pad_tag)

    test_text_sentences = executor1.test_dataset.text_sentences
    test_outfile = os.path.join(results_dir, "test.out.tsv")
    evaluate_pipeline(executor1, test_gold, test_predicted, test_text_sentences, test_outfile, "TEST", out_tags,
                      none_tag, pad_tag)


def generate_text_tags(dataset, evaluator):
    batch_size, seq_len = evaluator.predicted.shape
    text_tags = []
    for sent_index in range(batch_size):
        sent_tags = []
        for word_index in range(seq_len):
            if evaluator.mask[sent_index][word_index] == 0:
                sent_tags.append(dataset.text_tags[sent_index][word_index])
            else:
                sent_tags.append(dataset.out_tags[evaluator.predicted[sent_index][word_index]])
        text_tags.append(sent_tags)
    return text_tags


def convert_to_indexed_tags(text_tags, tag_vocab):
    indexed_tags = []
    for text_tag in text_tags:
        indexed_tags.append([tag_vocab.index(curr_tag) for curr_tag in text_tag])
    return np.array(indexed_tags)


def evaluate_pipeline(executor, gold, predicted, corpus, outfile, prefix, out_tags, none_tag, pad_tag):
    evaluator = Evaluator(gold=gold, predicted=predicted, tags=out_tags, ignore_tags=[none_tag, pad_tag],
                          none_tag=none_tag, pad_tag=pad_tag)

    print("{0}: Entity-Level Metrics:".format(prefix))
    print(evaluator.entity_metric.report())
    print("{0}: Token-Level (Without 'O' Tag) Metrics:".format(prefix))
    print(evaluator.significant_token_metric.report())

    executor.print_outputs(corpus=corpus, gold=gold, predicted=predicted, mapping=out_tags, outfile=outfile)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Binary CNN-LSTM-CRF Model for Sequence Labeling")
    ap.add_argument("--name1", type=str, default="binary-cnn-lstm-crf",
                    help="model name to be run first (Default: 'binary-cnn-lstm-crf')")
    ap.add_argument("--name2", type=str, default="cnn-cosine",
                    help="model name to be run second (on output of first) (Default: 'cnn-cosine')")
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
    ap.add_argument("--pipeline_dir", type=str, default="pipeline",
                    help="path to pipeline output directory, relative to data directory (Default: 'pipeline')")
    ap.add_argument("--train_path", type=str, default="train.tsv",
                    help="path to train dataset (train.tsv|std_train.tsv|jnlpba_train.tsv) (Default: 'train.tsv')")
    ap.add_argument("--dev_path", type=str, default="dev.tsv",
                    help="path to dev dataset (dev.tsv|std_dev.tsv|jnlpba_dev.tsv) (Default: 'dev.tsv')")
    ap.add_argument("--test_path", type=str, default="test.tsv",
                    help="path to test dataset (test.tsv|std_test.tsv|jnlpba_test.tsv) (Default: 'test.tsv')")
    ap.add_argument("--word_vocab_path", type=str, default="glove_vocab.txt",
                    help="path to word vocab (Default: 'glove_vocab.txt')")
    ap.add_argument("--tags_path1", type=str, default="tag_vocab.txt",
                    help="path to output tags vocab for 1st model. Use 'tag_vocab.txt' for full tags vocab. "
                         "Use 'std_tag_vocab.txt' for standard 5 tags vocab. "
                         "Use 'jnlpba_tag_vocab.tsv' for exact (5-tag) settings used by MTL-BioInformatics-2016 "
                         "(ref: https://github.com/cambridgeltl/MTL-Bioinformatics-2016)"
                         "Use 'out_freq_tag_vocab.txt' for reduced tags, when considering input tags information. "
                         "(Default: 'tag_vocab.txt')")
    ap.add_argument("--tags_path2", type=str, default="tag_vocab_two_step.txt",
                    help="path to output tags vocab for 2nd model. Use 'tag_vocab.txt' for full tags vocab. "
                         "Use 'std_tag_vocab.txt' for standard 5 tags vocab. "
                         "Use 'jnlpba_tag_vocab.tsv' for exact (5-tag) settings used by MTL-BioInformatics-2016 "
                         "(ref: https://github.com/cambridgeltl/MTL-Bioinformatics-2016)"
                         "Use 'out_freq_tag_vocab.txt' for reduced tags, when considering input tags information. "
                         "(Default: 'tag_vocab_two_step.txt')")
    ap.add_argument("--inp_tag_vocab_path", type=str, default="empty_inp_tag_vocab.txt",
                    help="path to input tags vocab. Use 'empty_inp_tag_vocab.txt' if don't want to use tag info. "
                         "Use 'inp_freq_tag_vocab.txt' for specifying default input tag info."
                         "(Default: 'empty_inp_tag_vocab.txt')")
    ap.add_argument("--mask_tag_vocab_path", type=str, default="mask_freq_tag_vocab.txt",
                    help="path to masked tags vocab. Use 'empty_mask_tag_vocab.txt' if don't want to mask out any tag. "
                         "Use 'mask_freq_tag_vocab.txt' for masking tags which are very less frequent in the dataset, "
                         "including some special tags that need to be separately handled."
                         "(Default: 'mask_freq_tag_vocab.txt')")
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
                         "(jnlpba_tag_w2v_emb.txt|jnlpba_tag_use_emb.txt|jnlpba_tag_full_emb.txt) "
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
    ap.add_argument("--use_maxpool1", action="store_true", help="model1: max pool over CNN output to get char "
                                                                "embeddings, else does concatenation (Default: False)")
    ap.add_argument("--use_maxpool2", action="store_true", help="model2: max pool over CNN output to get char "
                                                                "embeddings, else does concatenation (Default: False)")
    ap.add_argument("--use_pos_tag", action="store_true", help="embed POS tag information (Default: False)")
    ap.add_argument("--use_dep_tag", action="store_true", help="embed dep-parse tag information (Default: False)")
    ap.add_argument("--use_tag_cosine_sim1", action="store_true",
                    help="model1: compute cosine sim with tag embeddings as additional layer in model (Default: False)")
    ap.add_argument("--use_tag_cosine_sim2", action="store_true",
                    help="model2: compute cosine sim with tag embeddings as additional layer in model (Default: False)")
    ap.add_argument("--kernel_size", type=int, default=5, help="kernel size for CNN (Default: 5)")
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
                    help="type information (none/self/window/pretrained) (Default: 'none')")
    ap.add_argument("--use_tfo", type=str, default="none",
                    help="use transformer (may not use LSTM then). 'simple' creates a basic tfo. "
                         "'xl' uses TransformerXL model layer. (none|simple|xl) (Default: 'none')")
    ap.add_argument("--loss_type", type=str, default="cosine",
                    help="loss function (cross_entropy/cosine) (Default: cosine)")
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
    ap.add_argument("--lr", type=float, default=0.001, help="learning rate (Default: 0.001)")
    ap.add_argument("--seed", type=int, default=42, help="manual seed for reproducibility (Default: 42)")
    ap.add_argument("--fine_tune_bert", action="store_true", help="fine-tune bert embeddings (Default: False)")
    ap.add_argument("--use_cpu", action="store_true", help="force CPU usage (Default: False)")
    ap.add_argument("--no_eval_print", action="store_true",
                    help="don't output verbose evaluation matrices (Default: False)")
    ap = ap.parse_args()
    main(ap)
