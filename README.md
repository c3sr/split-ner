# NER on Security Data

Simple NER model framework designed with the target of low-resource extractions on security data.
## Installation
```commandline
pip install -r requirements.txt
pip install -e .
```
## Training
In corresponding config.json, do:
 ```json
{
 "do_train": true,
 "eval": null
}
```
Then,
```commandline
cd secner
CUDA_VISIBLE_DEVICES=0,1 python main.py --config config.json
```
#### Base Models
The base model can be specified in ```base_model``` field of config.json. Supported models:
* BERT-Base: ```bert-base-uncased```
* BioBERT-Base: ```dmis-lab/biobert-base-cased-v1.1```
* SciBERT-Base: ```allenai/scibert_scivocab_uncased```
## Evaluation
For evaluating on saved checkpoint (say, ```4840```), in config.json, do:
```json
{
 "do_train": false,
 "eval": "4840"
}
```

## Results

| Test Entity F1(%)| BioNLP13CG | JNLPBA | CoNLL03 | OntoNotes | GENIA |
|--|--|--|--|--|--|
QA3-BioBERT | 86.02 | 74.52 | IBM | | | |
QA4-BioBERT | 86.17 |       | | | | |
BioBERT | 85.64 | 73.75*      | | | | |
BioBERT-CRF | 85.69 |       | | | | |
SciBERT | 85.77 | 74.08*      | | | | |
QA4-BioBERT-CharCNN | 85.73**** |       | | | | |
QA4-MentionMask-BioBERT | IBM |       | | | | |
QA4-MentionMask-BioBERT-CharCNN | IBM |       | | | | |
KGQA |  | Running (79.24 reported)      | | | | |
CNN-LSTM-CRF | 84.1 |      76.2 | 91.6 | 86.4 | 72.4|
MRC |  |     78.85***  | 93.04** |  | |
SOTA | 89.58 (KGQA with all datasets) |      79.24 (KGQA with all datasets) | 94.3 (LUKE) | 92.07 (MRC-Dice) | -|

```
* on older evaluation code
** reported training logs in their codebase
*** on development set
**** 32: char embedding size, 16: filters, 0.1: dropout, 5: kernel size
```
