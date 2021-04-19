import logging
import spacy
from datasets import load_dataset
from spacy.tokens import Doc
from tqdm import tqdm

from secner.utils.general import setup_logging, generate_aux_tag_vocab_from_data

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    conll03_dataset = load_dataset("conll2003")
    process(conll03_dataset, dataset_key="train", out_file="train")
    process(conll03_dataset, dataset_key="validation", out_file="dev")
    process(conll03_dataset, dataset_key="test", out_file="test")


def process(conll03_dataset, dataset_key, out_file):
    logger.info("data: {0}".format(dataset_key))
    nlp = spacy.load("en_core_web_sm")
    tokenizer_map = dict()
    nlp.tokenizer = lambda x: Doc(nlp.vocab, tokenizer_map[x])

    pos_tags = conll03_dataset[dataset_key].features["pos_tags"].feature.names
    ner_tags = conll03_dataset[dataset_key].features["ner_tags"].feature.names
    processed_dataset = []

    for sent_index in tqdm(range(len(conll03_dataset[dataset_key]))):
        sent = conll03_dataset[dataset_key][sent_index]
        n = len(sent["tokens"])
        text = " ".join(sent["tokens"])
        tokenizer_map[text] = sent["tokens"]
        doc = nlp(text)
        processed_sent = []
        for token_index in range(n):
            entry = [sent["tokens"][token_index],
                     pos_tags[sent["pos_tags"][token_index]],
                     doc[token_index].dep_,
                     ner_tags[sent["ner_tags"][token_index]]]
            processed_sent.append("\t".join(entry))
        processed_dataset.append("\n".join(processed_sent))

    with open("../../data/conll/{0}.tsv".format(out_file), "w", encoding="utf-8") as f:
        f.write("\n\n".join(processed_dataset) + "\n")

    generate_aux_tag_vocab_from_data("conll")


if __name__ == "__main__":
    main()
