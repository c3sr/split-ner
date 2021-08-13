import argparse
import os
import sys
from gensim import models
import argparse

if __name__=='__main__':

        parser = argparse.ArgumentParser()
        parser.add_argument("-i", "--input_file", type=str, default=None, help="Input clean text file")
        parser.add_argument("-o", "--output_model", type=str, help="Output word embedding model")
        args = parser.parse_args()


        file_name=args.input_file
        fr=open(file_name, 'r')
        lines=fr.readlines()
        fr.close()

        sentences=[]

        for line in lines:
                strs=line.split()
                sentences.append(strs)

        print("num of sentences = "+str(len(sentences)))
        model=models.word2vec.Word2Vec(size=50, window=3, min_count = 3, sg=1, hs=0, negative=5, workers=20)
        #sample=1e-3, sg=1, hs=1, negative=0,

        model.build_vocab(sentences)
        words = model.wv.vocab.keys()
        print("vocab size", len(words))
        pattern_vocab_file = open("pattern_vocab.txt", "w")

        pattern_vocab_file.write("PAD\n")
        for pattern in words:
            pattern_vocab_file.write(pattern+"\n")
        pattern_vocab_file.write("UNK\n")
        pattern_vocab_file.close()

        model.train(sentences, total_examples=len(sentences), epochs=30)
        model.save(args.output_model)

