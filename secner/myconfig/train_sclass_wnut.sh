#! /bin/bash

data_root='/shared/data/jatin2/sec-ner/data'
out_root='/shared/data/jatin2/sec-ner/multirun'

train_epochs=10
seq_length=512

data="onto_final"
base_model="roberta-base"
eval_step=2000

