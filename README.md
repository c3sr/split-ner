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

| Test Entity F1(%)                        | BioNLP13CG                  | JNLPBA         |   CoNLL                    | Genia     | Onto                 |
|------------------------------------------|-----------------------------|----------------|----------------------------|-----------|----------------------|
| BERT-Base                                | 81.940                      |                |                            |           |                      |
| BioBERT                                  | 85.644                      | 74.35          |  90.932                    |           |                      |
| SciBERT                                  | **86.092**                  | 74.68          |                            |           |                      |
| SciBERT-PairedSentence                   | 85.606*                     |                |                            |           |                      |
| BioBERT-BO-Tagging                       | 85.218                      |                |                            |           |                      |
| BioBERT-BO-Tagging-CE_Weighted           | 85.502                      |                |                            |           |                      |
| BioBERT-CE_Weighted                      | 85.575                      |                |                            |           |                      |
| BioBERT-Dice                             | 85.966                      | 74.829         |  90.504                    |           |                      |
| BioBERT-CRF                              | 85.685                      |                |                            |           |                      |
| CharCNN1                                 | 50.283                      |                |                            |           |                      |
| BioBERT-CharCNN1-Pattern0LSTM            | 85.048                      |                |                            |           |                      |
| BioBERT-CharCNN5-Pattern0LSTM            | 85.915                      |                |                            |           |                      |
| BioBERT-CharCNN5                         | 85.836                      |                |                            |           |                      |
| BioBERT-CharCNN5-Pattern1LSTM            |**86.064**                   | 73.516         |  90.571                    |           |                      |
| BioBERT-Pattern1LSTM                     | 85.385                      |                |                            |           |                      |
| BioBERT-CharCNN7-Pattern0LSTM            | 85.773                      |                |                            |           |                      |
| BioBERT-CharCNN5-Pattern0LSTM2           | 85.777                      |                |                            |           |                      |
| BioBERT-CharCNN9-Pattern0LSTM            | 85.857                      |                |                            |           |                      |
| BioBERT-Punctuation                      | **86.348**                  | 73.844         |  **91.263**                |           |                      |
| BioBERT-Punctuation-WordType             | 85.648                      |                |                            |           |                      |
| BioBERT-Punctuation-CRF                  | 85.925                      |                |  90.993                    |           |                      |
| BioBERT-WordType                         | **86.166**                  |                |  90.772                    |           |                      |
| BioBERT-QA3                              | **86.023**                  | 74.52          |                            |           |                      |
| BioBERT-QA4                              | **86.172**                  | 74.499         |  90.954                    |           |                      |
| BioBERT-QA4-Punctuation                  | **86.167**                  |                |                            |           |                      |
| BioBERT-QA4-Dice                         | 75.323*                     | 73.232*        |                            |           |                      |
| BioBERT-QA4-CRF                          | 84.950                      |                |                            |           |                      |
| BioBERT-QA4-EndCNN                       | -                           |                |                            |           |                      |
| BioBERT-QA4-CharCNN1                     | 85.731                      |                |                            |           |                      |
| BioBERT-QA4-CharCNN5-Pattern0LSTM        | 85.550*                     |                |                            |           |                      |
| BioBERT-QA4-CharCNN1-Highway             | **86.042**                  |                |                            |           |                      |
| BioBERT-QA4-Mention                      | 81.537                      |                |  **91.655**                |           |                      |
| BioBERT-QA3-Mention-CharCNN              | -                           |                |                            |           |                      |
| CNN-LSTM-CRF                             | 84.1                        | **76.2**       |  **91.6**                  | 72.4      | 86.4                 |
| SOTA                                     | 89.58(KGQA)                 | 79.24(KGQA)    |  94.3(LUKE), 93.4(Cloze)   |   -       | 92.07 (MRC-Dice)     |

```
* trained for 50 epochs (others are trained for 300 epochs)
For CoNLL, OntoNotes: BERT, BioBERT, SciBERT -> all correspond to general English BERT model
```

## Precision / Recall Analysis
Precision, Recall distribution for some good performing models to understand where we can still improve upon. The values are calculated from predictions file created for the test set. Becase of the different between BERT-based tokenization and actual sentence tokenization, the results for models are not same as in the table above, but they correspond to the same model.

### BioNLP13CG Corpus

| Model                                    | Precision              | Recall                 | Micro F1               |
|------------------------------------------|------------------------|------------------------|------------------------|
| BioBERT                                  | 86.1666                | 85.8160                | 85.9910                |
| BioBERT-Punctuation                      | 87.6171                | 85.6562                | 86.6255                |
| BioBERT-Dice                             | 86.6764                | 86.0340                | 86.3540                |
| BioBERT-CharCNN5-Pattern1LSTM            | 86.8093                | 86.0776                | 86.4419                |
| BioBERT-WordType                         | 86.3524                | 86.5281                | 86.4402                |
| BioBERT-QA4                              | 88.6159                | 84.3918                | 86.4523                |
| BioBERT-QA4-Punctuation                  | 88.0719                | 84.7697                | 86.3892                |
| BioBERT-QA4-CharCNN1-Highway             | 88.2282                | 84.7406                | 86.4492                |

1. QA models have greater precision than normal models but lower recall.
2. CharCNN+Pattern helped increase recall.
3. Dice Loss is performing as almost as good as CharCNN+Pattern case.
4. In the normal training case, nothing is giving as good a precision as normal Punctuation handling.
5. In normal training, WordType and CharCNN+Pattern, increased both precision and recall.
6. In QA, Punctuation reduced precision (and increased recall).
7. WordType gave the highest recall in the normal case till now.