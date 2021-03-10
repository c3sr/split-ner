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
| SciBERT-PairedSentence                   | 85.606**                    |                |                            |           |                      |
| BioBERT-BO-Tagging                       | 85.218                      |                |                            |           |                      |
| BioBERT-BO-Tagging-CE_Weighted           | 85.502                      |                |                            |           |                      |
| BioBERT-CE_Weighted                      | 85.575                      |                |                            |           |                      |
| BioBERT-HeadToken                        | **86.175**                  | 73.649         |  **91.487** (-MISC: 92.930 PER: 96.898)|         |            |
| BioBERT-POS-Tag                          | 85.900                      |                |                            |           |                      |
| BioBERT-DEP-Tag                          | 85.799                      |                |                            |           |                      |
| BioBERT-Dice                             | 85.966                      | 74.829         |  90.504                    |           |                      |
| BioBERT-CRF                              | 85.685                      |                |                            |           |                      |
| CharCNN1                                 | 50.283                      |                |                            |           |                      |
| BioBERT-CharCNN1-Pattern0LSTM            | 85.048                      |                |                            |           |                      |
| BioBERT-CharCNN5-Pattern1LSTM            |**86.064**                   | 73.516         |  90.571                    |           |                      |
| BioBERT-CharCNN5-Pattern2LSTM*           | 85.588                      |                |                            |           |                      |
| BioBERT-CharCNN5                         | 85.836                      |                |                            |           |                      |
| BioBERT-CharCNN5-Pattern1LSTM-SubText    |**86.025**                   |                |                            |           |                      |
| BioBERT-Pattern1LSTM                     | 85.385                      |                |                            |           |                      |
| BioBERT-CharCNN7-Pattern0LSTM            | 85.773                      |                |                            |           |                      |
| BioBERT-CharCNN5-Pattern0LSTM2           | 85.777                      |                |                            |           |                      |
| BioBERT-CharCNN9-Pattern0LSTM            | 85.857                      |                |                            |           |                      |
| BioBERT-FlairLSTM*                       | 85.626                      |                |                            |           |                      |
| BioBERT-FlairLSTM-Pattern1LSTM*          | 85.570                      |                |                            |           |                      |
| BioBERT-FlairLSTM-Pattern2LSTM*          | 85.323                      |                |                            |           |                      |
| BioBERT-Punctuation                      | **86.348**                  | 73.844         |  91.263                    |           |                      |
| BioBERT-Punctuation-LongContext300       | 85.532                      |                |                            |           |                      |
| BioBERT-Punctuation-Warmup               | 85.839                      |                |                            |           |                      |
| BioBERT-PunctuationExtended              | **86.037**                  |                |  90.657                    |           |                      |
| BioBERT-Punctuation-HeadToken            | **86.357**                  |                |  **91.681**                |           |                      |
| BioBERT-Punctuation-WordType             | 85.648                      |                |                            |           |                      |
| BioBERT-Punctuation-CRF                  | **86.049**                  |                |  90.993                    |           |                      |
| BioBERT-Punctuation-And                  | **86.133**                  |                |                            |           |                      |
| BioBERT-Punctuation-CE-PunctLoss         | 85.725                      |                |                            |           |                      |
| BioBERT-WordType                         | **86.166**                  |                |  90.772                    |           |                      |
| BioBERT-WordType-SubText                 | **86.211**                  |                |  91.009                    |           |                      |
| BioBERT-QA3                              | **86.023**                  | 74.52          |                            |           |                      |
| BioBERT-QA4                              | **86.172**                  | 74.499         |  90.954                    |           |                      |
| BioBERT-QA4-Punctuation                  | **86.167**                  |                |                            |           |                      |
| BioBERT-QA4-WordType                     | 85.848                      |                |                            |           |                      |
| BioBERT-QA4-Dice                         | 75.323**                    | 73.232**       |                            |           |                      |
| BioBERT-QA4-CRF                          | 84.950                      |                |                            |           |                      |
| BioBERT-QA4-EndCNN                       | -                           |                |                            |           |                      |
| BioBERT-QA4-CharCNN1                     | 85.731                      |                |                            |           |                      |
| BioBERT-QA4-CharCNN5-Pattern0LSTM        | 85.550**                    |                |                            |           |                      |
| BioBERT-QA4-CharCNN1-Highway             | **86.042**                  |                |                            |           |                      |
| BioBERT-QA4-Mention                      | 81.537                      |                |  **91.655**                |           |                      |
| BioBERT-QA3-Mention-CharCNN              | -                           |                |                            |           |                      |
| CNN-LSTM-CRF                             | 84.1                        | **76.2**       |  **91.6**                  | 72.4      | 86.4                 |
| SOTA                                     | 89.58(KGQA)                 | 79.24(KGQA)    |  94.3(LUKE), 93.4(Cloze)   |   -       | 92.07 (MRC-Dice)     |

```
* new runs
** trained for 50 epochs (others are trained for 300 epochs)
For CoNLL, OntoNotes: BERT, BioBERT, SciBERT -> all correspond to general English BERT model
Note: DiceLoss and PunctLoss helped improve a lot on DEV set but did not improve on the TEST set
```

## Precision / Recall Analysis
Precision, Recall distribution for some good performing models to understand where we can still improve upon. The values are calculated from predictions file created for the test set. Becase of the different between BERT-based tokenization and actual sentence tokenization, the results for models are not same as in the table above, but they correspond to the same model.

### BioNLP13CG Corpus

| Model                                    | Precision              | Recall                 | Micro F1               |
|------------------------------------------|------------------------|------------------------|------------------------|
| BioBERT                                  | 86.1666                | 85.8160                | 85.9910                |
| BioBERT-Punctuation                      | 87.6171                | 85.6562                | 86.6255                |
| BioBERT-PunctuationExtended              | 86.1328                | 86.7461                | 86.4383                |
| BioBERT-Dice                             | 86.6764                | 86.0340                | 86.3540                |
| BioBERT-CharCNN5-Pattern1LSTM            | 86.8093                | 86.0776                | 86.4419                |
| BioBERT-WordType                         | 86.3524                | 86.5281                | 86.4402                |
| BioBERT-QA4                              | 88.6159                | 84.3918                | 86.4523                |
| BioBERT-QA4-Punctuation                  | 88.0719                | 84.7697                | 86.3892                |
| BioBERT-QA4-WordType                     | 88.4910                | 84.0285                | 86.2020                |
| BioBERT-QA4-CharCNN1-Highway             | 88.2282                | 84.7406                | 86.4492                |

1. QA models have greater precision than normal models but lower recall.
2. CharCNN+Pattern helped increase recall.
3. Dice Loss is performing as almost as good as CharCNN+Pattern case.
4. In the normal training case, nothing is giving as good a precision as normal Punctuation handling.
5. Extended punctuation handling in Normal case gave us the best recall till now, but compromised on precision.
6. In normal training, WordType and CharCNN+Pattern, increased both precision and recall.
7. In QA, Punctuation reduced precision (and increased recall).
8. WordType gave the highest recall in the normal case till now.

### Class-Wise Analysis

| Model | Count | Precision | Recall | Micro F1 |
|-------|-------|-----------|--------|----------|
| Overall | - | 87.6171 | 85.6562 | 86.6255 |
| Cancer | 924 | 91.4989 | 88.5281 | 89.9890 |
| Simple_chemical | 727 | 86.6154 | 77.4415 | 81.7720 |
| Gene_or_gene_product | 2520 | 90.7022 | 91.7460 | 91.2211 |
| Organ | 156 | 75.3247 | 74.3590 | 74.8387 |
| Organism | 518 | 90.1381 | 88.2239 | 89.1707 |
| Multi-tissue_structure | 303 | 78.8194 | 74.9175 | 76.8189 |
| Cell | 1013 | 90.0901 | 88.8450 | 89.4632 |
| Cellular_component | 180 | 85.5556 | 85.5556 | 85.5556 |
| Organism_substance | 102 | 94.1860 | 79.4118 | 86.1702 |
| Tissue | 184 | 63.0631 | 76.0870 | 68.9655 |
| Amino_acid | 62 | 89.2857 | 40.3226 | 55.5556 |
| Immaterial_anatomical_entity | 31 | 61.7647 | 67.7419 | 64.6154 |
| Organism_subdivision | 39 | 63.1579 | 61.5385 | 62.3377 |
| Pathological_formation | 88 | 60.2564 | 53.4091 | 56.6265 |
| Developing_anatomical_structure | 17 | 44.4444 | 47.0588 | 45.7143 |
| Anatomical_system | 17 | 50.0000 | 5.8824 | 10.5263 |

### Pre-Clustering

Experimented with clustering mentions for high-resource entities to handle diversity issues. Accouringly, divided **Gene_or_gene_product** into 4 sub-classes and trained a model. Finally in the test set, the resultant labels in the 4 sub-classes were remaped to the parent class and precision/recall were measured. For training, we used the QA setup and worked with only 3 output classes (Gene_or_gene_product, Simple_chemical, Cancer). However, this approach is not found to give improvements over the original setup.

| BioNLP13CG (QA model) Without Clustering | Precision              | Recall                 | F1                     |
|------------------------------------------|------------------------|------------------------|------------------------|
| Overall                                  | 88.6535                | 87.2932                | 87.9681                |
| Gene_or_gene_product                     | 88.4540                | 91.5079                | 89.9591                |
| Simple_chemical                          | 87.8489                | 73.5903                | 80.0898                |
| Cancer                                   | 89.7868                | 86.5801                | 88.1543                |

| BioNLP13CG (QA model) With Clustering    | Precision              | Recall                 | Micro F1               |
|------------------------------------------|------------------------|------------------------|------------------------|
| Overall                                  | 88.4203                | 84.9640                | 86.6577                |
| Gene_or_gene_product                     | **88.6556**            | 87.1775                | 87.9103                |
| Simple_chemical                          | 86.1538                | **77.0289**            | **81.3362**            |
| Cancer                                   | 89.4318                | 85.1732                | 87.2505                |

### New Considerations

1. BioNLP13CG dataset has nested entities (around 1%) which should be considered.
2. BioNLP13CG dataset has paragraphical texts. Considering this might give more information to the model and facilitate NER.
3. Paragrapical structure can help identify key phrases which can help in boundary detection issues.
