import argparse
import os
import xml.etree.ElementTree as ET
from _collections import defaultdict

from sklearn.model_selection import train_test_split

from src.utils.corpus_utils import add_linguistic_features


class Token:
    def __init__(self, start, text, tag, pos_tag=None, dep_tag=None):
        self.start = start
        self.text = text
        self.tag = tag
        self.pos_tag = pos_tag
        self.dep_tag = dep_tag

    def __str__(self):
        return "({0}, {1}, {2}, {3}, {4})".format(self.start, self.text, self.tag, self.pos_tag, self.dep_tag)

    def __repr__(self):
        return "({0}, {1}, {2}, {3}, {4})".format(self.start, self.text, self.tag, self.pos_tag, self.dep_tag)


def begin_tag(tag):
    return "B-{0}".format(tag)


def intermediate_tag(tag):
    return "I-{0}".format(tag)


def update_tags(tokens, entity):
    entity_start = entity.start
    entity_end = entity.start + len(entity.text)
    matched_count = 0
    for token in tokens:
        if entity_start <= token.start < entity_end:
            if matched_count == 0:
                token.tag = begin_tag(entity.tag)
            else:
                token.tag = intermediate_tag(entity.tag)
            matched_count += 1
    return tokens


def get_leaf_entities(node):
    tag_entity = "cons"

    prefix = ""
    if node.text:
        prefix += node.text

    entities = []
    is_leaf = True
    # TODO: Capture this node's entity "cons" tag as well, for nested entity parsing
    for child in node:
        is_leaf = False
        child_entities, child_text = get_leaf_entities(child)
        for ce in child_entities:
            ce.start += len(prefix)
            entities.append(ce)
        prefix += child_text
        if child.tail:
            prefix += child.tail

    if is_leaf and node.tag == tag_entity:
        # TODO: currently ignoring leaf nodes which do not have a "sem" tag. Ideally, to handle these cases,
        #  their parent "sem" tags define their logic
        if "sem" in node.attrib:
            entities.append(Token(start=0, text=prefix, tag=node.attrib["sem"]))
        else:
            global missed_leaf_tags
            missed_leaf_tags += 1

    return entities, prefix


def get_tokens(node):
    tag_token = "w"
    out_tag_other = "O"
    prefix_length = 0
    if node.text:
        prefix_length += len(node.text)
    tokens = []
    for child in node:
        if child.tag == tag_token:
            pos_tag = child.attrib["c"]
            if len(pos_tag) == 0:
                pos_tag = "O"  # replace empty POS tag with a special "O" tag
            tokens.append(Token(start=prefix_length, text=child.text, tag=out_tag_other, pos_tag=pos_tag))
        prefix_length += len(child.text)
        if child.tail:
            prefix_length += len(child.tail)
    return tokens


def process_sentence(sent_term, sent_pos):
    tokens = get_tokens(sent_pos)
    entities, _ = get_leaf_entities(sent_term)
    for entity in entities:
        tokens = update_tags(tokens, entity)
    return tokens


def process_article(article_term, article_pos):
    tag_sent = ".//sentence"
    sentences_term = list(article_term.findall(tag_sent))
    sentences_pos = list(article_pos.findall(tag_sent))

    processed_sentences = []
    for st, sp in zip(sentences_term, sentences_pos):
        processed_sentences.append(process_sentence(st, sp))
    return processed_sentences


def process_corpus(corpus_term, corpus_pos):
    tag_article = ".//article"
    articles_term = list(corpus_term.findall(tag_article))
    articles_pos = list(corpus_pos.findall(tag_article))

    processed_articles = []
    for at, ap in zip(articles_term, articles_pos):
        processed_articles.extend(process_article(at, ap))

    add_linguistic_features(processed_articles, dep=True)
    return processed_articles


def write_to_file(samples, filepath):
    with open(filepath, "w") as f:
        for sentence in samples:
            for token in sentence:
                f.write("{0}\t{1}\t{2}\t{3}\n".format(token.text, token.pos_tag, token.dep_tag, token.tag))
            f.write("\n")


def generate_tags_vocab(corpus, vocabpath):
    tags_vocab = set()
    for sentence in corpus:
        for token in sentence:
            tags_vocab.add(token.tag)

    tags_vocab = sorted(list(tags_vocab))
    with open(vocabpath, "w") as f:
        for tag in tags_vocab:
            f.write("{0}\n".format(tag))


def generate_all_tags_vocab(corpus, vocabpath):
    tags_vocab = set()
    for sentence in corpus:
        for token in sentence:
            tags_vocab.add(token.tag)

    # add special tags
    tags_vocab.add("<PAD>")
    tags_vocab.add("<INFO>")
    tags_vocab.add("<MASK>")
    tags_vocab.add("<START>")
    tags_vocab.add("<STOP>")

    tags_vocab = sorted(list(tags_vocab))
    with open(vocabpath, "w") as f:
        for tag in tags_vocab:
            f.write("{0}\n".format(tag))


def generate_pos_tags_vocab(corpus, vocabpath):
    tags_vocab = set()
    for sentence in corpus:
        for token in sentence:
            tags_vocab.add(token.pos_tag)

    tags_vocab = sorted(list(tags_vocab))
    with open(vocabpath, "w") as f:
        for tag in tags_vocab:
            f.write("{0}\n".format(tag))


def generate_dep_tags_vocab(corpus, vocabpath):
    tags_vocab = set()
    for sentence in corpus:
        for token in sentence:
            tags_vocab.add(token.dep_tag)

    tags_vocab = sorted(list(tags_vocab))
    with open(vocabpath, "w") as f:
        for tag in tags_vocab:
            f.write("{0}\n".format(tag))


def generate_segregated_inp_out_tags_based_on_frequency(corpus, inp_tag_vocab_path, out_tag_vocab_path, threshold):
    tags_vocab = defaultdict(int)
    out_tags = set()
    inp_tags = set()
    for sentence in corpus:
        for token in sentence:
            tag = token.tag
            if tag.startswith("B-") or tag.startswith("I-"):
                tag = tag[2:]
                # count in both B/I forms of the tag (because we either take both as input tags or both as output tags)
                tags_vocab["B-{0}".format(tag)] += 1
                tags_vocab["I-{0}".format(tag)] += 1
            else:
                # is a non-B/I tag (like, "O" tag), so retain it in output tags
                out_tags.add(tag)

    for tag, freq in tags_vocab.items():
        if freq >= threshold:
            inp_tags.add(tag)
        else:
            out_tags.add(tag)

    inp_tags = sorted(list(inp_tags))
    out_tags = sorted(list(out_tags))

    with open(inp_tag_vocab_path, "w") as f:
        for tag in inp_tags:
            f.write("{0}\n".format(tag))

    with open(out_tag_vocab_path, "w") as f:
        for tag in out_tags:
            f.write("{0}\n".format(tag))


def generate_text_corpus_for_tag_embeddings(corpus, out_text_file, out_vocab_file):
    text_corpus = []
    vocab = set()
    for sentence in corpus:
        new_sentence = []
        for token in sentence:
            if token.tag.startswith("B-"):
                # just maintain one tag for n-gram tagged tokens
                # no difference between B-Tag and I-Tag
                entry = token.tag[2:]
            elif token.tag.startswith("I-"):
                continue
            else:
                # retain words which are not tagged
                entry = token.text.lower()
            new_sentence.append(entry)
            vocab.add(entry)
        text_corpus.append(new_sentence)

    with open(out_text_file, "w") as f:
        for sentence in text_corpus:
            f.write("{0}\n".format(" ".join(sentence)))

    vocab = sorted(list(vocab))
    with open(out_vocab_file, "w") as f:
        for word in vocab:
            f.write("{0}\n".format(word))


def read_tsv_corpus(corpus_path, delimiter="\t"):
    corpus = []
    with open(corpus_path, "r") as f:
        sentence = []
        start = 0
        for line in f:
            line = line.strip()
            if line and not line.startswith("-DOCSTART-"):
                s = line.split(delimiter)
                if len(s) == 2:
                    # 2 column format
                    sentence.append(Token(start=start, text=s[0], tag=s[1]))
                elif len(s) == 4:
                    # 4 column format
                    sentence.append(Token(start=start, text=s[0], pos_tag=s[1], dep_tag=s[2], tag=s[3]))
                else:
                    # print out line as a silent exception
                    print("inconsistent tokenization of line: [{0}]".format(line))
                start += len(s[0]) + 1  # to account for white space
            elif len(sentence) > 0:
                corpus.append(sentence)
                sentence = []
                start = 0

    return corpus


def rectify_corpus_issues(corpus):
    tokenizer_map = dict()
    for sentence in corpus:
        sentence_text = " ".join(token.text for token in sentence)
        if sentence_text not in tokenizer_map:
            tokenizer_map[sentence_text] = sentence
        else:
            # duplicate found. Linguistic features must match, else report.
            existing_sentence = tokenizer_map[sentence_text]
            match = check_tags(existing_sentence, sentence)
            if not match:
                print("Full Sentence: {0}".format(sentence_text))


def check_tags(sentence1, sentence2):
    match = True
    for token1, token2 in zip(sentence1, sentence2):
        if token1.pos_tag != token2.pos_tag:
            print("POS mismatch: {0}, {1}".format(token1, token2))
            match = False
        if token1.dep_tag != token2.dep_tag:
            print("DEP mismatch: {0}, {1}".format(token1, token2))
            match = False
        if token1.tag != token2.tag:
            print("NER mismatch: {0}, {1}".format(token1, token2))
            match = False
    return match


def update_corpus(new_corpus_map, existing_corpus_path):
    existing_corpus = read_tsv_corpus(existing_corpus_path)

    for index in range(len(existing_corpus)):
        existing_sentence = existing_corpus[index]
        key = generate_corpus_map_key(existing_sentence)
        existing_corpus[index] = new_corpus_map[key]

    write_to_file(existing_corpus, existing_corpus_path)


def generate_corpus_map_key(sentence):
    return ", ".join(["({0}, {1})".format(token.text, token.tag) for token in sentence])


def standardize_tags(corpus):
    std_corpus = []
    for sentence in corpus:
        std_corpus.append([Token(start=token.start, text=token.text, tag=get_std_tag(token.tag), pos_tag=token.pos_tag,
                                 dep_tag=token.dep_tag) for token in sentence])
    return std_corpus


def get_std_tag(tag_text):
    allowed_tags = ["protein", "DNA", "RNA", "cell_type", "cell_line"]
    if tag_text.startswith("B-") or tag_text.startswith("I-"):
        actual_tag = tag_text[4:]  # B-G#Tag
        for root_tag in allowed_tags:
            if actual_tag.startswith(root_tag):
                return "{0}G#{1}".format(tag_text[:2], root_tag)
        return "O"  # all non-allowed tags are mapped to "O" tag
    return tag_text  # special tags are returned as is


def get_cleaned_data_with_pos_dep(corpus_path, dep=True, pos=True, delimiter="\t"):
    corpus = read_tsv_corpus(corpus_path, delimiter=delimiter)
    corpus = add_linguistic_features(corpus, pos=pos, dep=dep)
    return corpus


def tag_to_text_genia(tag):
    if tag == "(AND G#cell_type G#cell_type)":
        return "composite cell type"
    if tag == "(AND G#protein_complex G#protein_complex)":
        return "composite protein complex"
    return " ".join(tag[2:].split("_"))  # line: G#<tag name>


def main(args):
    corpus_term = ET.parse(args.inp_term).getroot()
    corpus_pos = ET.parse(args.inp_pos).getroot()
    corpus = process_corpus(corpus_term, corpus_pos)

    generate_tags_vocab(corpus, os.path.join(args.out, "tag_vocab.txt"))
    generate_all_tags_vocab(corpus, os.path.join(args.out, "all_tag_vocab.txt"))

    inp_tag_vocab_path = os.path.join(args.out, "inp_freq_tag_vocab.txt")
    out_tag_vocab_path = os.path.join(args.out, "out_freq_tag_vocab.txt")
    generate_segregated_inp_out_tags_based_on_frequency(corpus, inp_tag_vocab_path, out_tag_vocab_path, args.threshold)

    # Train/Dev/Test: 0.6/0.2/0.2 split
    corpus_non_test, corpus_test = train_test_split(corpus, test_size=0.2)
    corpus_train, corpus_dev = train_test_split(corpus_non_test, test_size=0.25)

    write_to_file(corpus, os.path.join(args.out, "corpus.tsv"))
    write_to_file(corpus_train, os.path.join(args.out, "train.tsv"))
    write_to_file(corpus_dev, os.path.join(args.out, "dev.tsv"))
    write_to_file(corpus_test, os.path.join(args.out, "test.tsv"))

    emb_text_corpus_path = os.path.join(args.out, "emb_text_corpus.txt")
    emb_vocab_path = os.path.join(args.out, "emb_vocab.txt")
    corpus_train = read_tsv_corpus(os.path.join(args.out, "train.tsv"))
    generate_text_corpus_for_tag_embeddings(corpus_train, emb_text_corpus_path, emb_vocab_path)

    # utility to update old corpus with POS and DEP tags
    # corpus = read_tsv_corpus(os.path.join(args.out, "corpus.tsv"))
    # corpus_map = {generate_corpus_map_key(sentence): sentence for sentence in corpus}
    # update_corpus(corpus_map, os.path.join(args.out, "train.tsv"))
    # update_corpus(corpus_map, os.path.join(args.out, "dev.tsv"))
    # update_corpus(corpus_map, os.path.join(args.out, "test.tsv"))

    rectify_corpus_issues(corpus)

    corpus = read_tsv_corpus(os.path.join(args.out, "corpus.tsv"))
    generate_pos_tags_vocab(corpus, os.path.join(args.out, "pos_tag_vocab.txt"))
    generate_dep_tags_vocab(corpus, os.path.join(args.out, "dep_tag_vocab.txt"))

    # 5-class standard benchmark corpus creation
    corpus = read_tsv_corpus(os.path.join(args.out, "corpus.tsv"))
    std_corpus = standardize_tags(corpus)
    generate_tags_vocab(std_corpus, os.path.join(args.out, "std_tag_vocab.txt"))
    generate_all_tags_vocab(std_corpus, os.path.join(args.out, "std_all_tag_vocab.txt"))

    # Train/Dev/Test: 0.81/0.09/0.1 split
    std_corpus_non_test, std_corpus_test = train_test_split(std_corpus, test_size=0.1, random_state=args.seed)
    std_corpus_train, std_corpus_dev = train_test_split(std_corpus_non_test, test_size=0.1, random_state=args.seed)

    write_to_file(std_corpus, os.path.join(args.out, "std_corpus.tsv"))
    write_to_file(std_corpus_train, os.path.join(args.out, "std_train.tsv"))
    write_to_file(std_corpus_dev, os.path.join(args.out, "std_dev.tsv"))
    write_to_file(std_corpus_test, os.path.join(args.out, "std_test.tsv"))

    std_emb_text_corpus_path = os.path.join(args.out, "std_emb_text_corpus.txt")
    std_emb_vocab_path = os.path.join(args.out, "std_emb_vocab.txt")
    std_corpus_train = read_tsv_corpus(os.path.join(args.out, "std_train.tsv"))
    generate_text_corpus_for_tag_embeddings(std_corpus_train, std_emb_text_corpus_path, std_emb_vocab_path)

    jnlpba_corpus_train = get_cleaned_data_with_pos_dep(os.path.join(args.jnlpba_root, args.jnlpba_train_path))
    jnlpba_corpus_dev = get_cleaned_data_with_pos_dep(os.path.join(args.jnlpba_root, args.jnlpba_dev_path))
    jnlpba_corpus_test = get_cleaned_data_with_pos_dep(os.path.join(args.jnlpba_root, args.jnlpba_test_path))

    write_to_file(jnlpba_corpus_train, os.path.join(args.out, "jnlpba_train.tsv"))
    write_to_file(jnlpba_corpus_dev, os.path.join(args.out, "jnlpba_dev.tsv"))
    write_to_file(jnlpba_corpus_test, os.path.join(args.out, "jnlpba_test.tsv"))

    jnlpba_corpus_train = read_tsv_corpus(os.path.join(args.out, "jnlpba_train.tsv"))
    generate_tags_vocab(jnlpba_corpus_train, os.path.join(args.out, "jnlpba_tag_vocab.txt"))
    generate_all_tags_vocab(jnlpba_corpus_train, os.path.join(args.out, "jnlpba_all_tag_vocab.txt"))
    generate_pos_tags_vocab(jnlpba_corpus_train, os.path.join(args.out, "jnlpba_pos_tag_vocab.txt"))
    generate_dep_tags_vocab(jnlpba_corpus_train, os.path.join(args.out, "jnlpba_dep_tag_vocab.txt"))

    jnlpba_emb_text_corpus_path = os.path.join(args.out, "jnlpba_emb_text_corpus.txt")
    jnlpba_emb_vocab_path = os.path.join(args.out, "jnlpba_emb_vocab.txt")
    jnlpba_corpus_train = read_tsv_corpus(os.path.join(args.out, "jnlpba_train.tsv"))
    generate_text_corpus_for_tag_embeddings(jnlpba_corpus_train, jnlpba_emb_text_corpus_path, jnlpba_emb_vocab_path)


if __name__ == "__main__":
    # corpus link: http://www.geniaproject.org/genia-corpus/term-corpus
    ap = argparse.ArgumentParser(description="GENIA term(entity) corpus parser. Converts to tsv form.")
    ap.add_argument("-it", "--inp_term", default="../../../GENIA_term_3.02/GENIAcorpus3.02.xml", type=str,
                    help="raw xml ner term corpus location (Default: '../../../GENIA_term_3.02/GENIAcorpus3.02.xml')")
    ap.add_argument("-ip", "--inp_pos", default="../../../GENIA_term_3.02/GENIAcorpus3.02.pos.xml", type=str,
                    help="raw xml pos, token corpus location (Default: '../../../GENIA_term_3.02/GENIAcorpus3.02.xml')")
    ap.add_argument("-o", "--out", default="../../../GENIA_term_3.02/", type=str,
                    help="processed corpus output location (Default: '../../../GENIA_term_3.02/')")
    ap.add_argument("--jnlpba_root", default="../../../MTL-Bioinformatics-2016/data/JNLPBA", type=str,
                    help="input corpus root (Default: '../../../MTL-Bioinformatics-2016/data/JNLPBA')")
    ap.add_argument("--jnlpba_train_path", default="train.tsv", type=str,
                    help="train corpus input location, relative to corpus root (Default: 'train.tsv')")
    ap.add_argument("--jnlpba_dev_path", default="devel.tsv", type=str,
                    help="dev corpus input location, relative to corpus root (Default: 'devel.tsv')")
    ap.add_argument("--jnlpba_test_path", default="test.tsv", type=str,
                    help="test corpus input location, relative to corpus root (Default: 'test.tsv')")
    ap.add_argument("-t", "--threshold", default=6000, type=int,
                    help="frequency threshold for segregating input/output tags. This helps in feeding surrounding "
                         "input-tags information which may help in detecting the output-tags (Default: 4000)")
    ap.add_argument("-s", "--seed", default=42, type=int,
                    help="Random seed for corpus train/dev/test split (Default: 42)")
    ap = ap.parse_args()

    missed_leaf_tags = 0
    main(ap)
    print("processing done")
    print("missed leaf tags (requires complex nested tag-name inference): {0}".format(missed_leaf_tags))
