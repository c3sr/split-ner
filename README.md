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

| Test Entity F1(%)            | BioNLP13CG                  | JNLPBA       | CoNLL | OntoNotes | Genia |
|------------------------------|-----------------------------|--------------|-------|-----------|-------|
| BERT-Base                    | 81.940                      |              |       |           |       |
| BioBERT                      | 85.644                      | 74.35        |       |           |       |
| SciBERT                      | 86.092                      | 74.68        |       |           |       |
| SciBERT-PairedSentence       | 85.606*                     |              |       |           |       |
| BioBERT-BO-Tagging           | 85.218                      |              |       |           |       |
| BioBERT-CE_Weighted          | 85.575                      |              |       |           |       |
| BioBERT-Dice                 | 85.966                      |              |       |           |       |
| BioBERT-CRF                  | 85.685                      |              |       |           |       |
| CharCNN1                     | 50.283                      |              |       |           |       |
| BioBERT-CharCNN1-PatternLSTM | 85.048                      |              |       |           |       |
| BioBERT-CharCNN5-PatternLSTM | 85.915                      |              |       |           |       |
| BioBERT-CharCNN9-PatternLSTM | running                     |              |       |           |       |
| BioBERT-Punctuation          | 86.348                      |              |       |           |       |
| BioBERT-QA3                  | 86.023                      | 74.52        |       |           |       |
| BioBERT-QA4                  | 86.172                      | 74.499       |       |           |       |
| BioBERT-QA4-CRF              | 84.950                      |              |       |           |       |
| BioBERT-QA4-EndCNN           | -                           |              |       |           |       |
| BioBERT-QA4-CharCNN1         | 85.731                      |              |       |           |       |
| BioBERT-QA4-BiDAF            | 86.042                      |              |       |           |       |
| BioBERT-QA4-Mention          | 81.389**                    |              |       |           |       |
| BioBERT-QA4-Mention-CharCNN  | -                           |              |       |           |       |
| CNN-LSTM-CRF                 | 84.1                        | 76.2         |       |           |       |
| SOTA                         | 89.58(KGQA with extra data) | 79.24 (KGQA) |       |           |       |


| Test Entity F1(%)| BioNLP13CG | JNLPBA | CoNLL03 | OntoNotes | GENIA |
|--|--|--|--|--|--|
QA3-BioBERT | 86.02 | 74.52 | IBM | | | |
QA4-BioBERT | 86.17 |       | | | | |
BioBERT | 85.64 | 74.35      | | | | |
BioBERT-CRF | 85.98 |       | | | | |
SciBERT | 85.77 | 74.68     | | | | |
QA4-BioBERT-CharCNN | 85.73*** |       | | | | |
QA4-MentionMask-BioBERT | IBM |       | | | | |
QA4-MentionMask-BioBERT-CharCNN | IBM |       | | | | |
KGQA |  | Running (79.24 reported)      | | | | |
CNN-LSTM-CRF | 84.1 |      76.2 | 91.6 | 86.4 | 72.4|
MRC |  |     78.85**  | 93.04* |  | |
SOTA | 89.58 (KGQA with all datasets) |      79.24 (KGQA with all datasets) | 94.3 (LUKE) | 92.07 (MRC-Dice) | -|

```
* reported training logs in their codebase
** on development set
*** 32: char embedding size, 16: filters, 0.1: dropout, 5: kernel size
```
