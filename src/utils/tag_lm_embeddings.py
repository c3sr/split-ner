import argparse
import os

import numpy as np
import torch
from transformers import BertTokenizer, BertModel

from src.datasets.genia import tag_to_text_genia
from src.utils.general import parse_emb_file


def read_tags(tag_file, tag_to_text_fn):
    tag_texts = dict()
    with open(tag_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # TODO: may give unique representation to B-/I- tags in plain text like,
            #  "beginning/continuation of protein molecule"
            if line.startswith("B-") or line.startswith("I-"):
                line = line[2:]
                tag_texts[line] = tag_to_text_fn(line)
            elif line == "O":
                tag_texts[line] = "other"
            elif line == "<PAD>":
                tag_texts[line] = "padding"
            elif line == "<INFO>":
                tag_texts[line] = "entity"  # or, 'biomedical entity'
            elif line == "<START>":
                tag_texts[line] = "start of sentence"
            elif line == "<STOP>":
                tag_texts[line] = "end of sentence"
            elif line == "<MASK>":
                tag_texts[line] = "mask"

    print("For generation of tag embeddings using pre-trained language models, using tag-to-text mapping:")
    print("Tag,Text")
    for tag in tag_texts:
        print("{0},{1}".format(tag, tag_texts[tag]))

    return tag_texts


# generate universal sentence encoder embeddings
def generate_use_embeddings(tag_texts):
    import tensorflow_hub as hub

    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    sorted_tag_keys = sorted(list(tag_texts.keys()))
    tag_text_inputs = [tag_texts[key] for key in sorted_tag_keys]

    tag_embeddings = model(tag_text_inputs).numpy()

    emb_dict = dict()
    for index, key in enumerate(sorted_tag_keys):
        emb_dict[key] = tag_embeddings[index]

    return emb_dict


# generate BERT/SciBERT embeddings
def generate_bert_embeddings(tag_texts, bert_model_name="allenai/scibert_scivocab_uncased", from_tf=False):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    bert_model = BertModel.from_pretrained(bert_model_name, from_tf=from_tf).to(device)

    emb_dict = dict()
    for tag_key, tag_text in tag_texts.items():
        tag_indexed_tokens = bert_tokenizer(tag_text, return_tensors="pt")["input_ids"]
        tag_token_embeddings = bert_model(tag_indexed_tokens.to(device))[0].squeeze()
        emb_dict[tag_key] = torch.mean(tag_token_embeddings, dim=0).detach().cpu().numpy()

    return emb_dict


def concatenate_w2v_with_use(use_emb_file, w2v_emb_file, out_file):
    use_emb_dict = parse_emb_file(use_emb_file, has_header_line=False)
    w2v_emb_dict = parse_emb_file(w2v_emb_file, has_header_line=False)

    w2v_emb_dim = 0
    for key in w2v_emb_dict:
        w2v_emb_dim = len(w2v_emb_dict[key])
        break

    emb_dict = dict()
    for tag in use_emb_dict:
        if tag not in w2v_emb_dict:
            w2v_emb = [0.] * w2v_emb_dim
        else:
            w2v_emb = w2v_emb_dict[tag]
        emb_dict[tag] = np.array(use_emb_dict[tag] + w2v_emb)

    write_emb_to_file(emb_dict, out_file)


def write_emb_to_file(emb_dict, out_file):
    with open(out_file, "w") as f:
        for tag in sorted(emb_dict.keys()):
            print_tag = "_".join(tag.split(" "))
            f.write("{0} {1}\n".format(print_tag, " ".join([str(x) for x in emb_dict[tag]])))


def main(args):
    args.tag_vocab_file = os.path.join(args.root_dir, args.tag_vocab_file)
    args.use_tag_emb_file = os.path.join(args.root_dir, args.use_tag_emb_file)
    args.w2v_tag_emb_file = os.path.join(args.root_dir, args.w2v_tag_emb_file)
    args.bert_tag_emb_file = os.path.join(args.root_dir, args.bert_tag_emb_file)
    args.annotation_guideline_bert_tag_emb_file = os.path.join(args.root_dir,
                                                               args.annotation_guideline_bert_tag_emb_file)
    args.full_tag_emb_file = os.path.join(args.root_dir, args.full_tag_emb_file)

    # for default GENIA corpus
    tag_dict = read_tags(args.tag_vocab_file, tag_to_text_fn=tag_to_text_genia)

    # for general cases
    # tag_dict = read_tags(args.tag_vocab_file, tag_to_text_fn=lambda tag: " ".join(tag.split("_")))

    emb_dict = generate_use_embeddings(tag_dict)
    write_emb_to_file(emb_dict, args.use_tag_emb_file)

    emb_dict = generate_bert_embeddings(tag_texts=tag_dict, bert_model_name=args.bert_model, from_tf=args.from_tf)
    write_emb_to_file(emb_dict, args.bert_tag_emb_file)

    # for JNLPBA annotation guidelines-based tag embeddings (uncomment the following lines)
    # annotation_guideline_tag_dict = read_tags(args.tag_vocab_file,
    #                                           tag_to_text_fn=tag_to_text_annotation_guidelines_jnlpba)
    #
    # annotation_guideline_emb_dict = generate_bert_embeddings(tag_texts=annotation_guideline_tag_dict,
    #                                                          bert_model_name=args.bert_model, from_tf=args.from_tf)
    # write_emb_to_file(annotation_guideline_emb_dict, args.annotation_guideline_bert_tag_emb_file)

    concatenate_w2v_with_use(args.use_tag_emb_file, args.w2v_tag_emb_file, args.full_tag_emb_file)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Univ. Sentence Encoder (USE) embeddings for tags")

    ap.add_argument("--root_dir", type=str, default="../../data/GENIA_term_3.02",
                    help="tags vocab file (Default: '../../data/GENIA_term_3.02')")
    ap.add_argument("--tag_vocab_file", type=str, default="all_tag_vocab.txt",
                    help="tags vocab file, relative to root dir "
                         "(all_tag_vocab.txt|std_all_tag_vocab.txt|jnlpba_all_tag_vocab.txt) "
                         "(Default: 'all_tag_vocab.txt')")
    ap.add_argument("--use_tag_emb_file", type=str, default="tag_use_emb.txt",
                    help="USE tag emb file, relative to root dir "
                         "(tag_use_emb.txt|std_tag_use_emb.txt|jnlpba_tag_use_emb.txt) "
                         "(Default: 'tag_use_emb.txt')")
    ap.add_argument("--w2v_tag_emb_file", type=str, default="tag_w2v_emb.txt",
                    help="W2V tag emb file, relative to root dir "
                         "(tag_w2v_emb.txt|std_tag_w2v_emb.txt|jnlpba_tag_w2v_emb.txt) "
                         "(Default: 'tag_w2v_emb.txt')")
    ap.add_argument("--bert_tag_emb_file", type=str, default="tag_scibert_emb.txt",
                    help="BERT/SciBERT tag emb file, relative to root dir "
                         "(tag_scibert_emb.txt|std_tag_scibert_emb.txt|jnlpba_tag_scibert_emb.txt) "
                         "(Default: 'tag_scibert_emb.txt')")
    ap.add_argument("--annotation_guideline_bert_tag_emb_file", type=str,
                    default="jnlpba_tag_annotation_guideline_scibert_emb.txt",
                    help="BERT/SciBERT tag emb file, relative to root dir ""(tag_annotation_guideline_scibert_emb.txt"
                         "|std_tag_annotation_guideline_scibert_emb.txt"
                         "|jnlpba_tag_annotation_guideline_scibert_emb.txt) "
                         "(Default: 'jnlpba_tag_annotation_guideline_scibert_emb.txt')")
    ap.add_argument("--bert_model", type=str, default="allenai/scibert_scivocab_uncased",
                    help="BERT model name (allenai/scibert_scivocab_uncased|bert-base-uncased"
                         "|../../../resources/biobert_v1.1_pubmed|etc.)"
                         "(Default: 'allenai/scibert_scivocab_uncased')")
    ap.add_argument("--from_tf", action="store_true",
                    help="tag embedding generator model is a pretrained tensorflow model. Use 'True' for models like, "
                         "'../../../resources/biobert_v1.1_pubmed' (Default: False)")
    ap.add_argument("--full_tag_emb_file", type=str, default="tag_full_emb.txt",
                    help="W2V+USE (Full) tag emb file, relative to root dir "
                         "(tag_full_emb.txt|std_tag_full_emb.txt|jnlpba_tag_full_emb.txt) "
                         "(Default: 'tag_full_emb.txt')")
    ap = ap.parse_args()
    main(ap)
