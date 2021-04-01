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
| RoBERTa-Base*                            | -                           |                |  90.728                    |           |                      |
| BioBERT*                                 | 85.644                      | 74.35          |  90.932                    |           | 83.204 (LR:5e-5)     |
| BioBERT-2LayerClassifier*                | 85.681                      |                |                            |           |                      |
| BioBERT-Freeze (LR:0.005)                | 70.921 (ep:140)             | running        |  running                   |           | running              |
| BioBERT-Freeze-MainLSTM (LR:0.001)       | 81.435                      |                |                            |           |                      |
| BioBERT-Freeze-Punctuation (LR:0.001)    | 81.698                      |                |                            |           |                      |
| BioBERT-Freeze-QA4-MainLSTM (LR:0.001)   | 80.290                      |                |                            |           |                      |
| BioBERT-WithKnownSpans                   | 85.387                      |                |                            |           |                      |
| BioBERT-WithKnownSpansAndLabels          | 85.960                      |                |                            |           |                      |
| SciBERT                                  | **86.092**                  | 74.68          |                            |           |                      |
| SciBERT-PairedSentence                   | 85.606**                    |                |                            |           |                      |
| BioBERT-BO-Tagging                       | 85.218                      |                |                            |           |                      |
| BioBERT-BO-Tagging-CE_Weighted           | 85.502                      |                |                            |           |                      |
| BioBERT-CE_Weighted                      | 85.575                      |                |  running                   |           |                      |
| BioBERT-HeadToken                        | **86.175**                  | 73.649         |  **91.487** (-MISC: 92.930 PER: 96.898)|         |            |
| BioBERT-POS-Tag                          | 85.900                      |                |  91.126                    |           |                      |
| BioBERT-DEP-Tag                          | 85.799                      |                |  running                   |           |                      |
| BioBERT-Dice                             | 85.966                      | 74.829         |  90.504                    |           |                      |
| BioBERT-CRF                              | 85.685                      |                |                            |           |                      |
| CharCNN1                                 | 50.283                      |                |                            |           |                      |
| BioBERT-CharCNN1-Pattern0LSTM            | 85.048                      |                |                            |           |                      |
| BioBERT-CharCNN5-Pattern1LSTM            |**86.064**                   | 73.516         |  90.571                    |           |                      |
| BioBERT-CharCNN5-Pattern2LSTM            | 85.588                      |                |                            |           |                      |
| BioBERT-CharCNN5                         | 85.836                      |                |                            |           |                      |
| BioBERT-CharCNN5-Pattern1LSTM-SubText    |**86.025**                   |                |                            |           |                      |
| BioBERT-Pattern1LSTM                     | 85.385                      |                |                            |           |                      |
| BioBERT-CharCNN7-Pattern0LSTM            | 85.773                      |                |                            |           |                      |
| BioBERT-CharCNN5-Pattern0LSTM2           | 85.777                      |                |                            |           |                      |
| BioBERT-CharCNN9-Pattern0LSTM            | 85.857                      |                |                            |           |                      |
| BioBERT-FlairLSTM                        | 85.626                      |                |                            |           |                      |
| BioBERT-FlairLSTM-Pattern1LSTM           | 85.570                      |                |                            |           |                      |
| BioBERT-FlairLSTM-Pattern2LSTM           | 85.323                      |                |                            |           |                      |
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
| BioBERT-QA3                              | **86.023**                  | 74.52          |  running                   |           |                      |
| BioBERT-QA4                              | **86.172**                  | 74.499         |  90.954                    |           |                      |
| BioBERT-QA4-QuestionType2(Where)*        | **86.642**                  | 74.330         |  91.358                    |           |                      |
| RoBERTa-QA4*                             | -                           |                |  91.338                    |           |                      |
| BioBERT-QA4-Nested                       | 85.855                      |                |                            |           |                      |
| BioBERT-QA4-Punctuation                  | **86.167**                  |                |                            |           |                      |
| BioBERT-QA4-WordType                     | 85.848                      |                |                            |           |                      |
| BioBERT-QA4-Dice                         | 75.323**                    | 73.232**       |                            |           |                      |
| BioBERT-QA4-CRF                          | 84.950                      |                |                            |           |                      |
| BioBERT-QA4-EndCNN                       | -                           |                |                            |           |                      |
| BioBERT-QA4-CharCNN1                     | 85.731                      |                |                            |           |                      |
| BioBERT-QA4-CharCNN5-Pattern0LSTM        | 85.550**                    |                |                            |           |                      |
| BioBERT-QA4-CharCNN1-Highway             | **86.042**                  |                |                            |           |                      |
| BioBERT-QA4-Mention                      | 81.537                      |                |**91.655**(needs retraining)|           |                      |
| BioBERT-QA3-Mention-CharCNN              | -                           |                |                            |           |                      |
| CNN-LSTM-CRF                             | 84.1                        | **76.2**       |  **91.6**                  | 72.4      | 86.4                 |
| SOTA                                     | 89.58(KGQA)                 | 79.24(KGQA)    |  94.3(LUKE), 93.4(Cloze)   |   -       | 92.07 (MRC-Dice)     |

```
* new runs
** trained for 50 epochs (others are trained for 300 epochs)
For CoNLL, OntoNotes: BERT, BioBERT, SciBERT -> all correspond to general English BERT model
Note: DiceLoss and PunctLoss helped improve a lot on DEV set but did not improve on the TEST set
```

## Analysis of Error Cases
1. Boundary Detection: 
   a. Puntuation Symbols: Semantics is not well captured by pretrained BERT models, causing undesired breakage in detected entities (tried Punctuation vector to handle)
   b. Modifier prefix/suffix: adjectives which add to the entity are not detected by the model. Potentially this could be because of lack of similar cases in training set (needs to be checked) (tried adding POS/DEP vecs to handle this but did not help)
2. Out of Vocabulary Terms (tried pattern/char CNN to handle)

Since, punctuation symbols cause issues, tried removing punctuations from test set and directly testing a model (trained with punctuation symbols) and also separated trained a new model on non-punctuation data. Both the models did not perform as well as the original setting (with punctuations)

| Model Scenario                             | Test Micro-F1       |
|--------------------------------------------|---------------------|
| BioBERT-PunctDataTrained-PunctDataEval     | 86.0055**           | 
| BioBERT-PunctDataTrained-NoPunctDataEval   | 80.0209             |
| BioBERT-NoPunctDataTrained-NoPunctDataEval | 80.6674             |

** There has been a recent small bug fix in ```token_type_id``` setting which may have made the results slightly better than previous reported value of ```85.9910```.

## Span Detection and Classification Perspective
Instead of Sequence Labeling and Question-Asnwering perspective, we look at a pipelined approach in which we (1) first detect spans using QA model then, (2) classify identified spans into ```K``` output classes (tags) again using QA model. Every detected span is classified into some tag (without ```B/I``` and no ```NONE``` tag).

1. Part 1 QA: What is the entity mentioned in the text ? <Sentence> . Training: Every span converted to B/I/E-Entity span and train 4-class QA model. Training data size does not hence increase at all.
2. Part 2 QA: <Sentence> . What is <mention span> ? Training: Gold spans are taken and all data is converted to this format. One training sample for each span in gold dataset. **Note**: We don't currently train the span classifier on **imperfect spans** but that can be added as well.

#### BioNLP13CG

| Model           | BERT-level Span Test Micro-F1                           | Dataset-level Span Test P | Dataset-level Span Test R | Dataset-level Span Test F1 |
|-----------------|---------------------------------------------------------|---------------------------|---------------------------|----------------------------|
| Span Detector   | 89.808                                                  | 90.6365                   | 89.6091                   | 90.1198                    |
| Span Classifier | 94.056                                                  | 94.0561                   | 94.0561                   | 94.0561                    |
| Pipeline        | 93.465 (for span class. on gen. output from span det.)  | 86.3296                   | 85.4627                   | **85.8940**                |

#### CoNLL

| Model           | BERT-level Span Test Micro-F1                           | Dataset-level Span Test P | Dataset-level Span Test R | Dataset-level Span Test F1 |
|-----------------|---------------------------------------------------------|---------------------------|---------------------------|----------------------------|
| Span Detector   | 94.991                                                  | 95.3300                   | 95.1376                   | 95.2337                    |
| Span Classifier | 95.149                                                  | 95.0835                   | 95.0835                   | 95.0835                    |
| Pipeline        | 94.795 (for span class. on gen. output from span det.)  | 91.4506                   | 91.8220                   | **91.6360**                |

#### JNLPBA

| Model           | BERT-level Span Test Micro-F1                           | Dataset-level Span Test P | Dataset-level Span Test R | Dataset-level Span Test F1 |
|-----------------|---------------------------------------------------------|---------------------------|---------------------------|----------------------------|
| Span Detector   | 77.753                                                  | 74.7685                   | 82.2835                   | 78.3462                    |
| Span Classifier | 94.528                                                  | 94.4682                   | 94.5229                   | 94.4956                    |
| Pipeline        | 94.009 (for span class. on gen. output from span det.)  | 71.1385                   | 79.3358                   | **75.0139**                |

#### Motivation
The core idea is BERT model finetuning with additional external cues concatenated as vectors or parallel CNN/LSTM's are possibly not getting trained well. So, why not use another BERT model itself and fine-tune it on some other desired sub-task to get that additional information learnt well (which we would have otherwise learnt from punctuation vec or CNN/LSTM).

The easiest way to break the problem into sub-problems is this span pipeline. Also note that in QA model, a single BERT model is rich enough to not just detect spans but classify them with high precision. Lower recall could be due to model's representative limitations. So, make the task easier. Train the model for only span detection task. The diversity of spans will be more now but the model should potentially be able to learn well. Then train another separate model for span classification. So, overall advantage is, we have more parameters and tweeking power for the overall NER task but downside is that approach is pipelined. So, error propagates across the stages of the pipeline.

## Precision / Recall Analysis
Precision, Recall distribution for some good performing models to understand where we can still improve upon. The values are calculated from predictions file created for the test set. Becase of the different between BERT-based tokenization and actual sentence tokenization, the results for models are not same as in the table above, but they correspond to the same model.

#### BioNLP13CG Corpus

| Model                                    | Precision              | Recall                 | Micro F1               |
|------------------------------------------|------------------------|------------------------|------------------------|
| BioBERT                                  | 86.1666                | 85.8160                | 85.9910                |
| BioBERT-Freeze                           | 75.5802                | 75.2507                | 75.4151                |
| BioBERT-Punctuation                      | 87.6171                | 85.6562                | 86.6255                |
| BioBERT-PunctuationExtended              | 86.1328                | **86.7461**            | 86.4383                |
| BioBERT-Dice                             | 86.6764                | 86.0340                | 86.3540                |
| BioBERT-HeadToken                        | 86.1878                | 86.1503                | 86.1690                |
| BioBERT-Punctuation-HeadToken            | 86.8460                | 85.8741                | 86.3573                |
| BioBERT-PosTag                           | 86.3849                | 85.8451                | 86.1141                |
| BioBERT-DepTag                           | 86.0954                | 86.2956                | 86.1954                |
| BioBERT-CharCNN5-Pattern1LSTM            | 86.8093                | 86.0776                | 86.4419                |
| BioBERT-WordType                         | 86.3524                | 86.5281                | 86.4402                |
| BioBERT-WordType-SubText                 | 86.3222                | 86.6734                | 86.4975                |
| BioBERT-CE-Wt                            | 85.9218                | 85.9468                | 85.9343                |
| BioBERT-CE-PunctWt                       | 86.2094                | 86.0340                | 86.1216                |
| BioBERT-CRF                              | 86.1728                | 86.2229                | 86.1979                |
| BioBERT-QA                               | todo                   | todo                   | todo                   |
| BioBERT-QA4                              | 88.6159                | 84.3918                | 86.4523                |
| BioBERT-QA4-QuestionType2(Where)         | **89.2091**            | 84.5807                | **86.8333**            |
| BioBERT-QA4 (Nested)                     | 87.4147                | 84.4318                | 85.8973                |
| BioBERT-QA4-Punctuation                  | 88.0719                | 84.7697                | 86.3892                |
| BioBERT-QA4-WordType                     | 88.4910                | 84.0285                | 86.2020                |
| BioBERT-QA4-CharCNN1-Highway             | 88.2282                | 84.7406                | 86.4492                |

#### CoNLL 2003 Corpus

| Model                                    | Precision              | Recall                 | Micro F1               |
|------------------------------------------|------------------------|------------------------|------------------------|
| BERT                                     | 91.2352                | 91.4777                | 91.3563                |
| BERT-Freeze                              | running                | running                | running                |
| BERT-HeadToken                           | 91.6741                | 91.3005                | 91.4869                |
| BERT-PosTag                              | 91.2890                | 91.6608                | 91.4745                |
| BERT-DepTag                              | running                | running                | running                |
| BERT-CE-Wt                               | running                | running                | running                |
| BERT-Punctuation                         | 91.7480                | 91.6017                | 91.6748                |
| BERT-Punctuation-HeadToken               | 91.6888                | 91.6726                | 91.6807                |
| BERT-WordType-SubText                    | 91.4442                | 91.6549                | 91.5494                |
| BERT-CharCNN5-Pattern1LSTM               | 91.1901                | 90.9639                | 91.0768                |
| BERT-Dice                                | 91.0485                | 90.4678                | 90.7572                |
| BERT-QA3                                 | running                | running                | running                |
| BERT-QA4                                 | 91.5402                | 90.8006                | 91.1689                |
| BERT-QA4-QuestionType2(Where)            | **92.4651**            | 91.1927                | **91.8245**            |

#### JNLPBA Corpus

| Model                                    | Precision              | Recall                 | Micro F1               |
|------------------------------------------|------------------------|------------------------|------------------------|
| BioBERT                                  | 70.9698                | 78.0651                | 74.3485                |
| BioBERT-Freeze                           | running                | running                | running                |
| BioBERT-Punctuation                      | 70.4206                | 78.4692                | 74.2274                |
| BioBERT-Dice                             | 72.0468                | 78.2267                | 75.0097                |
| BioBERT-QA                               | 72.0803                | 77.7559                | 74.8106                |
| BioBERT-QA4                              | -                      | -                      | - (todo: update here)  |
| BioBERT-QA4-QuestionType2(Where)         | 71.4543                | 78.1149                | 74.6363                |

#### OntoNotes Corpus

| Model                                    | Precision              | Recall                 | Micro F1               |
|------------------------------------------|------------------------|------------------------|------------------------|
| BioBERT                                  | 82.2538                | 84.5543                | 83.3882                |
| BioBERT-Freeze                           | running                | running                | running                |

1. QA models have greater precision than normal models but lower recall.
2. CharCNN+Pattern helped increase recall.
3. Dice Loss is performing as almost as good as CharCNN+Pattern case.
4. In the normal training case, nothing is giving as good a precision as normal Punctuation handling.
5. Extended punctuation handling in Normal case gave us the best recall till now, but compromised on precision.
6. In normal training, WordType and CharCNN+Pattern, increased both precision and recall.
7. In QA, Punctuation reduced precision (and increased recall).
8. WordType gave the highest recall in the normal case till now.
9. Simply changing the question from ```What``` to ```Where``` is the ```protein``` located in the text leads to performance improvement!

### Class-Wise Analysis

#### BioBERT-Punctuation model (Best Sequence Tagging Model)

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

#### BioBERT-QA4-QuestionType2(Where) model (Best QA Model)

| Overall | P: 89.2091 | R: 84.5807 | Micro F1: 86.8333 |
| --------|------------|------------|-------------------|
| Tag: Cancer | P: 91.9101 | R: 88.5281 | F1: 90.1874 |
| Tag: Simple_chemical | P: 87.6190 | R: 75.9285 | F1: 81.3559 |
| Tag: Gene_or_gene_product | P: 89.7887 | R: 91.0714 | F1: 90.4255 |
| Tag: Organ | P: 82.4818 | R: 72.4359 | F1: 77.1331 |
| Tag: Organism | P: 90.9836 | R: 85.7143 | F1: 88.2704 |
| Tag: Multi-tissue_structure | P: 85.0980 | R: 71.6172 | F1: 77.7778 |
| Tag: Cell | P: 93.3679 | R: 88.9437 | F1: 91.1021 |
| Tag: Cellular_component | P: 90.8537 | R: 82.7778 | F1: 86.6279 |
| Tag: Organism_substance | P: 97.4026 | R: 73.5294 | F1: 83.7989 |
| Tag: Tissue | P: 67.4757 | R: 75.5435 | F1: 71.2820 |
| Tag: Amino_acid | P: 82.1429 | R: 37.0968 | F1: 51.1111 |
| Tag: Immaterial_anatomical_entity | P: 72.4138 | R: 67.7419 | F1: 70.0000 |
| Tag: Organism_subdivision | P: 69.2308 | R: 46.1538 | F1: 55.3846 |
| Tag: Pathological_formation | P: 73.7705 | R: 51.1364 | F1: 60.4027 |
| Tag: Developing_anatomical_structure | P: 90.0000 | R: 52.9412 | F1: 66.6667 |
| Tag: Anatomical_system | P: 50.0000 | R: 5.8824 | F1: 10.5263 |

#### BioBERT-QA4 (Span Pipeline)

| Overall | P: 86.3296 | R: 85.4627 | Micro F1: 85.8940 |
|---------|------------|------------|-------------------|
| Tag: Cancer | P: 87.5403 | R: 88.2035 | F1: 87.8706 |
| Tag: Simple_chemical | P: 87.6006 | R: 75.1381 | F1: 80.8922 |
| Tag: Gene_or_gene_product | P: 88.8633 | R: 91.8983 | F1: 90.3553 |
| Tag: Organ | P: 73.5294 | R: 80.1282 | F1: 76.6871 |
| Tag: Organism | P: 88.8224 | R: 86.2403 | F1: 87.5123 |
| Tag: Multi-tissue_structure | P: 81.8182 | R: 74.5033 | F1: 77.9896 |
| Tag: Cell | P: 90.4330 | R: 88.6476 | F1: 89.5314 |
| Tag: Cellular_component | P: 80.1020 | R: 87.2222 | F1: 83.5106 |
| Tag: Organism_substance | P: 93.1818 | R: 80.3922 | F1: 86.3158 |
| Tag: Tissue | P: 56.4706 | R: 78.2609 | F1: 65.6036 |
| Tag: Anatomical_system | P: 66.6667 | R: 23.5294 | F1: 34.7826 |
| Tag: Immaterial_anatomical_entity | P: 62.0690 | R: 58.0645 | F1: 60.0000 |
| Tag: Organism_subdivision | P: 65.7143 | R: 58.9744 | F1: 62.1622 |
| Tag: Pathological_formation | P: 78.1250 | R: 56.8182 | F1: 65.7895 |
| Tag: Amino_acid | P: 84.0000 | R: 33.8710 | F1: 48.2759 |
| Tag: Developing_anatomical_structure | P: 80.0000 | R: 50.0000 | F1: 61.5385 |

### Pre-Clustering

Experimented with clustering mentions for high-resource entities to handle diversity issues. Accordingly, divided **Gene_or_gene_product** into 4 sub-classes and trained a model. Finally in the test set, the resultant labels in the 4 sub-classes were remaped to the parent class and precision/recall were measured. For training, we used the QA setup and worked with only 3 output classes (Gene_or_gene_product, Simple_chemical, Cancer). However, this approach is not found to give improvements over the original setup.

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

1. BioNLP13CG dataset has nested entities (around 1%) which should be considered. (**update:** our QA model gets almost equal performance)
2. BioNLP13CG dataset has paragraphical texts. Considering this might give more information to the model and facilitate NER. (**update:** naive implementation did not help. Can implement setup described in [this paper](https://arxiv.org/pdf/2006.01563v2.pdf).
3. Paragrapical structure can help identify key phrases which can help in boundary detection issues. (**update:** in next section)

### Span Guidance Analysis (Basically, any concatenated guidance analysis)

| Model                                    | Precision (Test)       | Recall (Test)          | Micro F1 (Test)        | Micro F1 (BERT subtext level)                       |
|------------------------------------------|------------------------|------------------------|------------------------|-----------------------------------------------------|
| BioBERT-Freeze          (LR:0.005)       | 75.5802                | 75.2507                | 75.4151                | (Train: 84.412, Dev: 71.059, Test: 70.921) (140 ep) |
| BioBERT-Freeze-GoldSpan (LR:0.005)       | 79.1092                | 79.2472                | 79.1782                | (Train: 87.207, Dev: 76.710, Test: 75.686) (20 ep)  |
| BioBERT-GoldSpan        (LR:1e-5)        | 86.2155                | 85.3510                | 85.7811                | (Train: 99.958, Dev: 85.647, Test: 85.387) (300 ep) |
| BioBERT-GoldSpan (LR:0.001 and 0.01)     | 0.0                    | 0.0                    | 0.0                    | -                                                   |

**Important Note:**
Ideally speaking, there is 4% (absolute) improvement with GoldSpan info. It should have shown in some form (to some slight extent at least) when trained with BioBERT, but it did not. Infact as seen below, even full gold-label info given in the input is not utilized properly when trained with BioBERT using contatenation with one-hot gold vectors.

### Training Concerns

We hypothesise that very low learning rate (1e-5) is not optimal for non-BERT params. So, overall, the BERT params get fine-tuned but the additional information given is not being learnt and utilized to its full potential.

**Experiment**: On giving actual true labels as one-hot embeddings to the model

| Model                                                     | Micro F1                    |
|-----------------------------------------------------------|-----------------------------|
| BioBERT-WithKnownSpansAndLabels (LR:1e-5)                 | 85.960 (Test)(**Striking!**)|
| BioBERT-Freeze-WithKnownSpansAndLabels (LR:1e-5)          | 54.502 (Dev)  (after 150 ep)|
| BioBERT-Freeze-WithKnownSpansAndLabels (LR:0.005)         | 100.000 (Test)              |

Also tried **fragmented training**, by training for first 15 epochs with BERT params frozen (LR: 0.005) and then fine-tuning complete thing (LR: 1e-5). But the second step gave 0.00 score consecutively for several epochs upon initiation of step 2, hence stopped.

Also tried high LR (0.01 and 0.001) training with BioBERT-GoldSpan (results for LR: 1e-5 in tables above), but for high LR it consistently gave 0.00 F1 for several epochs and hence, stopped.

**Interpretation of ```0.0``` in evaluation F1**: Typical case is that every token in the dataset is being tagged with the same tag (generally, something like ```I-Pathological_formation```. So, the span detector never detects any span and all gold spans are marked as unidentified. So, precision and recall are both calculated as ```0.0```.

**Question:**
Transformer training seems to be very sensitive to learning rate! We can't select a high LR, and low LR may not be able to train the remaining supporting parameters to train well. So, how to we strike a balance? Or, how else can we feed in additional guidance input to BERT-based transformer models?

#### Notes
1. Running QA/Normal models with BiLSTM as the main LM (with frozen BioBERT embeddings) (+Punctuation guidance) to see how well it performs and whether additional guidance vectors are utilized well. Also it seems, from training time perspective LSTM is **almost similar** to BERT (but has 2M params vs 102M in BERT).
