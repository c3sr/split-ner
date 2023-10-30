# SplitNER

This repository has the source code and pretrained models for our work, *Named Entity Recognition via Two Question-Answering-based Classifications* published in ACL 2023. 


We also have a video presentation of our work from ACL 2023 conference and some slides / poster. Do check them out in the [`resources`](resources) directory!

## Introduction

In this work, we address the NER problem by splitting it into two logical sub-tasks: 
1. **Span Detection** which simply extracts entity mention spans irrespective of entity type; 
2. **Span Classification** which classifies the spans into their entity types. 

Further, we formulate both sub-tasks as question-answering (QA) problems and produce two leaner models which can be optimized separately for each sub-task. Experiments with four cross-domain datasets demonstrate that this two-step approach is both effective and time efficient. Our system, SplitNER outperforms baselines on OntoNotes5.0, WNUT17 and a cybersecurity dataset and gives on-par performance on BioNLP13CG. In all cases, it achieves a significant reduction in training time compared to its QA baseline counterpart. The effectiveness of our system stems from fine-tuning the BERT model twice, separately for span detection and classification. 

## Installation

```shell script
pip install -r requirements.txt
pip install -e .
```

## Execution

All model training / evaluations happen via run arguments specified in config (JSON) files. In this repo, we also provide a small ```dummy``` dataset on which users can train and evaluate our model variants and baselines easily. Any new dataset should be formatted and used in the same way. 

The dummy data is actually a small randomly sampled set of sentences from ```WNUT17``` corpus [website](https://noisy-text.github.io/2017/emerging-rare-entities.html) and is just being used for illustration. As mentioned eariler, this should help the reader get the code up and runing, understand input/output data format and configure different train/eval scripts.

Dataset Citation:
```
Leon Derczynski, Eric Nichols, Marieke van Erp, Nut Limsopatham (2017) "Results of the WNUT2017 Shared Task on Novel and Emerging Entity Recognition", in Proceedings of the 3rd Workshop on Noisy, User-generated Text.
```

### Experimental Setup

* Every model has a `model_name` which is a unique identifier for the model architecture
* Every dataset should come as a separate `dataset_dir` under the `data` directory with all required files (see example `dummy` dataset dir)
* All model training experiments happen under the `experiment_dir` which by default is `out`
* To ensure replication of results, we make PyTorch environment deterministic with a `seed` specified in the config
* One may run many experiments with different seeds and then take average. Every such run is maintained separately by a `run_dir`
* When a `model_name` is trained on a `dataset_dir` in a specific run setting identified by `run_dir`, model checkpoints are created under `<repo root>/out/<dataset_dir>/<model_name>/<run_dir>`
* This above directory has `checkpoints` sub-directory which maintains PyTorch checkpoint snapshots for 2 models by default (latest and best so far). 
* It also has a `predictions` folder where outputs from the model when run in eval mode on `train`, `dev` or `test` sets is saved
* We have 3 different high-level model training scripts: 
  * `main.py`: trains using Sequence Tagging setup
  * `main_qa.py`: training using a Question-Answering setup
  * `main_span.py`: trains Span Classification model in Question-Answering setup
* For all major model variants and baselines, we specify train/eval configs ready to run on `dummy` dataset under the `config/dummy` directory
* All codes reside in the `<repo root>/splitner` directory which is configured as a python package (pip installable)
* For training a SplitNER model, we train two models independently:
  * Span Detector (`spandetect`)
  * Span Classifier using Dice Loss (`spanclass-dice`)
* After training both models, we generate output predictions from `spandetect` model and feed them as inputs to a trained `spanclass-dice` model to generate final output predictions

More details are provided in specific sections as detailed below.

### Training

The training script:
* Runs training for specified number of epochs
* Keeps logging training progress, best F1 metrics so far among other things
* Keeps evaluating on dev set after specified step interval and reports the F1 metrics
* At any given point of time, makes sure to save both best and latest checkpoints

Whenever training a model, make sure to set in its config file:
 ```json
{
 "do_train": true,
 "resume": null,
}
```

For resuming training from the latest checkpoint, find the last saved checkpoint from the checkpoint dir (default path: ```out/dummy/<model name>/<run name>/checkpoints```). Say last checkpoint was ```checkpoint-4840```, then update in config:
```json
{
 "resume": "4840",
}
```

### Evaluation

The training script itself, when run in eval mode:
* Reports the F1 metrics (token-level) for specified model checkpoint on test set
* Saves model predictions for train/dev/test sets under ```out/dummy/<model name>/<run name>/predictions```, by default

Whenever evaluating a model on saved checkpoint (say, ```4840```), make sure to set in its config file:
```json
{
 "do_train": false,
 "resume": "4840",
}
```

### Mention-Level Metrics

The following script takes the output predictions from the model and calculates mention-level metrics (most importantly Micro-F1 Score) as is most commonly reported in literature. If output predictions are in file ```<repo root>/out/dummy/<model name>/<run name>/predictions/test.tsv``` then,

```shell script
python analysis.py --experiment_dir out --dataset dummy --model <model name> --run_dir <run name> --file test
```

For getting metrics for span detection only, add flag ```--span_based```. Example, for ```spandetect``` model:

```shell script
python analysis.py --experiment_dir out --dataset dummy --model spandetect --run_dir <run name> --file test --span_based
```

## Run Commands

Move to the working directory
```shell script
cd splitner
```

### Span Detection Module

Make sure config file has ```"detect_spans": true```.  Example run command:

```shell script
CUDA_VISIBLE_DEVICES=0,1 python main_qa.py ../config/dummy/spandetect.json
```

### Span Classification Module

This uses Dice Loss, hence the name of the corresponding config script.

```shell script
CUDA_VISIBLE_DEVICES=0,1 python main_span.py ../config/dummy/spanclass-dice.json
```

### SplitNER Pipeline
Once the above models are individually trained and evaluated, copy ```test.tsv``` from ```predictions``` folder of Span Detector into ```predictions``` folder for Span Classification model. Rename this file as say, ```infer_inp.tsv```. Specify this file name in the config for span classifier (as shown below). Specify an output file path say, ```infer.tsv```.

```json
"infer_inp_path": "infer_inp.tsv",
"infer_out_path": "infer.tsv",
```

Run evaluation once again.

```shell script
CUDA_VISIBLE_DEVICES=0,1 python main_span.py ../config/dummy/spanclass-dice.json
```

The final outputs whill be in file ```infer.tsv``` under the ```predictions``` folder of Span Classification model. To get the mention-level F1-score metrics over your outputs, run:

```shell script
python analysis.py --experiment_dir out --dataset dummy --model spanclass-dice --run_dir <run name> --file infer
```

### Baselines

#### Single-QA

```shell script
CUDA_VISIBLE_DEVICES=0,1 python main_qa.py ../config/dummy/single-qa.json
```

#### Single-SeqTagging

```shell script
CUDA_VISIBLE_DEVICES=0,1 python main.py ../config/dummy/single-seqtag.json
```

### Ablations

#### Span Detection Module using Sequence Tagging Setup

```shell script
CUDA_VISIBLE_DEVICES=0,1 python main_qa.py ../config/dummy/spandetect-seqtag.json
```

#### Span Detection Module with No Character Pattern Features

```shell script
CUDA_VISIBLE_DEVICES=0,1 python main_qa.py ../config/dummy/spandetect-nocharpattern.json
```

#### Span Classification Module with Cross Entropy Loss

```shell script
CUDA_VISIBLE_DEVICES=0,1 python main_span.py ../config/dummy/spanclass-crossentropy.json
```

## Reproducing Results

Experiments are performed on 4 datsets: ```BioNLP13CG```, ```CyberThreats```, ```OntoNotes5.0```, ```WNUT17```. Config files are similar to the ones for ```dummy``` dataset with the below differences.

#### Max Sequence Length

* Set ```"max_seq_len": 512``` for ```OntoNotes5.0```
* Set ```"max_seq_len": 256``` for other datasets

#### Base Model

* Set ```"base_model": "bert-base-uncased"``` for ```WNUT17``` and ```CyberThreats```
* Set ```"base_model": "allenai/scibert_scivocab_uncased"``` for ```BIONLP13CG```
* Set ```"base_model": "roberta-base"``` and ```"model_mode": "roberta_std"``` for ```OntoNotes5.0```

### Pretrained Models

We also provide pretrained model checkpoints for all our model variants and baselines. Checkpoints are provided for all the publicly available datasets used in our study. Checkpoints are hosted on HuggingFace Hub under the [**Split-NER**](https://huggingface.co/splitner) classroom. To use a pretrained checkpoint, make sure to update the following fields in the config:

```json
{
 "do_train": false,
 "resume": null,
 "base_model": "<pretrained checkpoint name>",
}
```

The following table lists our pre-trained model checkpoint names:

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Dataset</th>
            <th>Pre-trained Checkpoint Name</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>Span Detection (QA)</td>
            <td>BioNLP13CG</td>
            <td><code>splitner/spandetect-qa-bionlp13cg</code></td>
        </tr>
        <tr>
            <td>OntoNotes5.0</td>
            <td><code>splitner/spandetect-qa-ontonotes</code></td>
        </tr>
        <tr>
            <td>WNUT17</td>
            <td><code>splitner/spandetect-qa-wnut17</code></td>
        </tr>
        <tr>
            <td rowspan=3>Span Classification (Dice Loss)</td>
            <td>BioNLP13CG</td>
            <td><code>splitner/spanclass-qa-dice-bionlp13cg</code></td>
        </tr>
        <tr>
            <td>OntoNotes5.0</td>
            <td><code>splitner/spanclass-qa-dice-ontonotes</code></td>
        </tr>
        <tr>
            <td>WNUT17</td>
            <td><code>splitner/spanclass-qa-dice-wnut17</code></td>
        </tr>
        <tr>
            <td rowspan=3>Span Detection Model<br>(Sequence Tagging)</td>
            <td>BioNLP13CG</td>
            <td><code>splitner/spandetect-seqtag-bionlp13cg</code></td>
        </tr>
        <tr>
            <td>OntoNotes5.0</td>
            <td><code>splitner/spandetect-seqtag-ontonotes</code></td>
        </tr>
        <tr>
            <td>WNUT17</td>
            <td><code>splitner/spandetect-seqtag-wnut17</code></td>
        </tr>
        <tr>
            <td rowspan=3>Span Detection<br>(QA - No Char Pattern)</td>
            <td>BioNLP13CG</td>
            <td><code>splitner/spandetect-qa-nocharpattern-bionlp13cg</code></td>
        </tr>
        <tr>
            <td>OntoNotes5.0</td>
            <td><code>splitner/spandetect-qa-nocharpattern-ontonotes</code></td>
        </tr>
        <tr>
            <td>WNUT17</td>
            <td><code>splitner/spandetect-qa-nocharpattern-wnut17</code></td>
        </tr>
        <tr>
            <td rowspan=3>Span Classification<br>(Cross-Entropy Loss)</td>
            <td>BioNLP13CG</td>
            <td><code>splitner/spanclass-qa-crossentropy-bionlp13cg</code></td>
        </tr>
        <tr>
            <td>OntoNotes5.0</td>
            <td><code>splitner/spanclass-qa-crossentropy-ontonotes</code></td>
        </tr>
        <tr>
            <td>WNUT17</td>
            <td><code>splitner/spanclass-qa-crossentropy-wnut17</code></td>
        </tr>
        <tr>
            <td rowspan=3>Single NER Model<br>(Sequence Tagging)</td>
            <td>BioNLP13CG</td>
            <td><code>splitner/single-seqtag-bionlp13cg</code></td>
        </tr>
        <tr>
            <td>OntoNotes5.0</td>
            <td><code>splitner/single-seqtag-ontonotes</code></td>
        </tr>
        <tr>
            <td>WNUT17</td>
            <td><code>splitner/single-seqtag-wnut17</code></td>
        </tr>
        <tr>
            <td rowspan=3>Single NER Model<br>(Question-Answering)</td>
            <td>BioNLP13CG</td>
            <td><code>splitner/single-qa-bionlp13cg</code></td>
        </tr>
        <tr>
            <td>OntoNotes5.0</td>
            <td><code>splitner/single-qa-ontonotes</code></td>
        </tr>
        <tr>
            <td>WNUT17</td>
            <td><code>splitner/single-qa-wnut17</code></td>
        </tr>
    </tbody>
</table>

Other ablations are only performed on the **BioNLP13CG** dataset and are listed below.

#### Char / Pattern Feature Ablation

| Model Name                                     | Pretrained Checkpoint Name                            |
|------------------------------------------------|-------------------------------------------------------|
| Span Detection (no Char / Pattern features)    | ```splitner/spandetect-qa-nocharpattern-bionlp13cg``` |
| Span Detection (Char features only)            | ```splitner/spandetect-qa-charonly-bionlp13cg```      |
| Span Detection (Pattern features only)         | ```splitner/spandetect-qa-patonly-bionlp13cg```       |
| Span Detection (Char + Pattern features)       | ```splitner/spandetect-qa-bionlp13cg```               |
| Span Detection (Char + Pattern + POS features) | ```splitner/spandetect-qa-pos-bionlp13cg```           |

#### Query Type Ablation

For ease of nomenclature we define query types as mentioned below.

| Query Type  | Description                                               |
|-------------|-----------------------------------------------------------|
| Q1          | *Extract important entity spans from the following text.* |
| Q2          | *Where is the entity mentioned in the text?*              |
| Q3          | *Find named entities in the following text.*              |
| Q4          | *< empty >*                                                 |

For the query types, the pre-trained model checkpoint names as as under:

| Model Name          | Pretrained Checkpoint Name                      |
|---------------------|-------------------------------------------------|
| Span Detection (Q1) | ```splitner/spandetect-qa-extract-bionlp13cg``` |
| Span Detection (Q2) | ```splitner/spandetect-qa-where-bionlp13cg```   |
| Span Detection (Q3) | ```splitner/spandetect-qa-find-bionlp13cg```    |
| Span Detection (Q4) | ```splitner/spandetect-qa-empty-bionlp13cg```   |

## Citation

We hope you like our work. 😊 Please do cite us when you reference Split-NER:
```
@inproceedings{arora2023split,
  title={Split-NER: Named Entity Recognition via Two Question-Answering-based Classifications},
  author={Arora, Jatin and Park, Youngja},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
  pages={416--426},
  year={2023}
}
```

## Correspondence
 
For any follow-ups, please don't hesitate in reaching out to us over e-mail!
```
Jatin Arora <jatinarora2702@gmail.com>
Youngja Park <young_park@us.ibm.com>
```

## Acknowledgements

This work was conducted under the IBM-Illinois Center for Cognitive Computing Systems Research ([C3SR](https://www.c3sr.com)), while the first author was an intern at IBM and a graduate student at University of Illinois Urbana-Champaign advised by [Prof. Jiawei Han](https://scholar.google.com/citations?user=Kv9AbjMAAAAJ&hl=en&oi=ao). We are very grateful to him for his continued guidance, support and valuable feedback throughout this work.
