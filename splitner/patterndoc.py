import os
import argparse
from splitner.dataset import NerDataset

def convert_data(args):
    pattern_doc_file = open(args.output, "w")

    with open(args.data, "r", encoding="utf-8") as f:
       sentence = "C"

       for line in f:
           line = line.strip()
           if line:
                row = line.split("\t")
                pattern = NerDataset.make_pattern(row[0], args.pattern_type)
                sentence += " "+pattern
           else:  
                sentence += " S"
                pattern_doc_file.write(sentence+"\n")
                sentence="C"

    pattern_doc_file.close()

    '''
    pattern_vocab_file = open(os.path.join(args.outpath, "pattern_vocab.txt"), "w")
    for pattern in pattern_set:
        pattern_vocab_file.write(pattern+"\n")
    pattern_vocab_file.close()
    '''

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pattern Vocab Generator")
    ap.add_argument("-data", default=None, help="training data file")
    ap.add_argument("-pattern_type", default="3", help="pattern_type")
    ap.add_argument("-output", default="./pattern_emb", help="output")
    args = ap.parse_args()

    convert_data(args)
