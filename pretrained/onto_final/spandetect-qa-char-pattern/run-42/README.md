### Span Detection Model (QA-Char-Pattern)

Question-Answering-based Span Detection model using additional character and pattern features.

#### Training
```

```

#### Evaluation
```

```

#### Micro-F1 Calculation
```
$ cd <REPO ROOT>/secner
$ python analysis.py \
    --experiment_dir pretrained \
    --dataset onto_final \
    --model spandetect-qa-char-pattern \
    --run_dir 42 \
    --file test \
    --span_based
```

| Overall | Cnt: 11243 | P: 92.1671 | R: 92.8311 | Micro F1: 92.4979 | 
|---------|------------|------------|------------|-------------------|
| Tag: ENTITY | Cnt: 11243 | P: 92.1671 | R: 92.8311 | F1: 92.4979 | 
