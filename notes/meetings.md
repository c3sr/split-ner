# Meeting Notes

**<span class="underline">2020/04/02 (Thursday) 4-5pm CT:</span>**

 

**Data Classification Toolkit**: Get client documents with sensitive personal information (PI). Such entities need to be extracted for document classification and redaction etc. The documents can come from client's box folders, employee laptop, GitHub etc.

 

**Data Format:** Work on unstructured data (text from documents), CSV (for tabular data), JSON, XML formats.

 

Have a **system pipeline** setup with the following components. The overall system could be provided on IBM Cloud:

 

1.      PDF parsing (preprocessing) (parses structure to get text with tables etc.)

 

2.      Linguistic Features (POS tags, NER, dependency parsing etc. over text)

 

3.      Entity Extraction (using pre-trained models for person names, orgs, addresses etc.) and regex (+custom logic on regex etc.), gazetteers etc. for non-trivial entities like SSN, Phone No., Passport Num etc.

 

4.      Entity Aggregator (for conflict resolution) - for identifying better spans of entities etc. (rule based weighted or voting based)

 

5.      Topic Classification Engine (ensemble models, 2-level classification, feature-based) - uses bag-of-words, POS, headings etc. for making features (LibSVM, UMass MALLET, DeepLearning4j etc. used for implementation). Example classes can be: project-planning doc, HR doc etc.

 

**Entity Extraction:**

 

The idea is to develop **deep learning model** that can do entity extraction using these **regex/patterns,** possibly utilizing distant supervision. Currently, in this domain, the following two are known to be popular:

 

1.      Distant supervision making use of external knowledge bases (like Wikidata) for relation extraction, or dictionaries for bio-medical domains

 

2.      Distant supervision with random forests as the model for classification

 

So, using regex patterns for weak (or distant) supervision signals and using this data in a deep learning setting is something that may be a promising direction to pursue.

**Bootstrapping** is another way to approach this problem. Use regex patterns to get some labelled data, train on it, use it to annotate more data and so on. But need to be aware of **semantic drift** issues. 

 

**<span class="underline">2020/05/18 (Monday) 1-2pm CT:</span>**

 

**Points for discussion:**

1.      Distant supervision is popular for relation extraction tasks using KBs like WikiData etc. Also popular for NER in bio-medical domain.

 

Some **keywords** that may help in literature survey:

o   Distant learning

o   Extraction in low resource settings

o   Pattern-based learning (or meta-patterns too)

 

2.      [Looking Beyond Label Noise: Shifted Label Distribution Matters in Distantly Supervised Relation Extraction](https://arxiv.org/abs/1904.09331) (EMNLP'19): In distantly supervised data creation, there may be a **label distribution bias** introduced which may affect model performance. So, may need to do some bias adjustments in the trained models.

 

3.      **Reinforcement learning** based techniques also seem to be popular for denoising data in distantly supervised settings.

 

4.      What **kind of named entity types** are there to be extracted?

·        General:

o   Person Name

o   Organization

o   Address

o   Religious Belief

o   Nationality

o   Gender

·        Numerical:

o   SSN

o   Passport Num

o   Driver's license

·        Health Related:

o   Medical condition (textual)

o   Hospital ID (numerical)

o   Treatment

o   Medicine names

·        Financial Domain:

o   Credit Card Num

o   Bank Acc. Num

·        Passwords

 

There can be other entities (both textual and numerical) but the idea is to develop a model that can generalize across all different entity types.

 

Some **example rules**:

 

For person name:

o   Length: 2-4 words

o   If more than 2 words, middle word is generally small

o   Are there systems that can identify names of different nationalities? (interesting!)

 

For organization names:

o   Can have sub-patterns within super-pattern. Ex. {IBM Research Cambridge} \[ORG\] -&gt; {IBM Research} \[ORG\] {Cambridge} \[CITY\]

o   The general model should be able to handle such cases as well (org within org etc.)

 

For SSN:

o   Is 10 digit long (3-3-4 format or 3.3.4 format)

o   First 3 numbers fall in this specified range

 

For  month / age: (entity disambiguation)

o   Both are 2 digit

o   Month: 1-12

o   Age: 1-99

  

**Context Definition**:

o   Text-intensive docs (**unstructured**),

·   Few sentences before and after the candidate entity.

·   Named Entity types of existing entities in the context neighborhood

·   Section header information may be useful for context (but this info may be inferred from neighborhood sentences as well)

 

o   CSV (tables), JSON, XML (**semi-structured**)

·   Column headers

·   Previous and next column headers or their types (can be inferred as well, if header not available)

 

**Inputs** to the model we are developing:

o   Different types of **context** for different kind of document inputs (described above)

o   Actual candidate entity **word** (from pattern matching etc.)

o   **Pattern** matched

o   **Internal sub-patterns** may also be useful (like org within org example above)

o   **Global document context** (Eg. Different countries may have different names for National ID but their pattern may be same, then language of document may help disambiguate)

 

The context may help in entity disambiguation**.** May think of a joint model that tries to disambiguate the domain based on the context and then tries to classify named entities.

 

**<span class="underline">2020/05/26 (Tuesday) 1-2pm CT:</span>**

**Project Updates:**

1.      Read some research papers (summarized below)

2.      Have setup LM-BiLSTM-CRF model (from [<span class="underline">GitHub</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__github.com_LiyuanLucasLiu_LM-2DLSTM-2DCRF&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=JkfjNag-L8gsRPaxJ1MFDT7wMDbUsluIAmzAd9pnUHg&e=)) and ran on [<span class="underline">CoNLL2003</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__github.com_synalp_NER_tree_master_corpus_CoNLL-2D2003&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=cYJmciTmfyz1ycYQbb-oa17BW8FLrH04lDC63z9CnzA&e=) dataset (achieving quoted accuracies ~98%)

3.      Found some blog posts, existing industry tools doing PII NER:

a.      [<span class="underline">LexNLP Library</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__www.groundai.com_project_lexnlp-2Dnatural-2Dlanguage-2Dprocessing-2Dand-2Dinformation-2Dextraction-2Dfor-2Dlegal-2Dand-2Dregulatory-2Dtexts_1&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=0tI9p63NI0IA28Fx4LBWGfi3fAuWm8ZIRj4bppzS_-I&e=) ([<span class="underline">GitHub</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__github.com_LexPredict_lexpredict-2Dlexnlp&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=_Ugkxn8LvtKVIvqplMi4HYI-rwvmQPyvM3_is2s4K7k&e=)): Information extraction from regulatory and legal text using StanfordCoreNLP, spacy, scikit-learn etc.

b.      [<span class="underline">Amazon Textract</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__aws.amazon.com_blogs_machine-2Dlearning_automatically-2Dextract-2Dtext-2Dand-2Dstructured-2Ddata-2Dfrom-2Ddocuments-2Dwith-2Damazon-2Dtextract_&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=wUlkH4jV8pF_YGR6OnZq1ahwiQlxBInqihaSgD9YnPc&e=): Extract PII from documents

c.      [<span class="underline">Microsoft Azure</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__docs.microsoft.com_en-2Dus_azure_search_cognitive-2Dsearch-2Dskill-2Dpii-2Ddetection&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=atn8SVAqlYbbTc_uPJtIDhGUht8D6jkLeA-rKD2tCdM&e=): PII detection from supplied raw text

d.      [<span class="underline">Extract</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__www.extractsystems.com_automated-2Ddata-2Dredaction-2Dsoftware_&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=E9ceh8IuPwMGrmHD1TxReFZn-QNJP0dnPS-qW6g3zVQ&e=): A proprietary PII redaction software in market

e.      [<span class="underline">Intuit</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__medium.com_blueprint-2Dby-2Dintuit_learning-2Dinformation-2Dextraction-2Dfrom-2Dstructured-2Ddocuments-2Dusing-2Dsynthetic-2Ddata-2Dand-2Dconditional-2Df1a9c7dac269&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=02YVqCJqIm1T3O8TjSahZ5WBNwTWe-5352o9CiiyTLg&e=): NER from documents (using CRF) over synthetic OCR-requiring image data. Use Word2Vec, LDA etc.

f.       [<span class="underline">Blog</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__towardsdatascience.com_deep-2Dlearning-2Dfor-2Dspecific-2Dinformation-2Dextraction-2Dfrom-2Dunstructured-2Dtexts-2D12c5b9dceada&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=wHc-lWZFrLOJ2IkUx3AI6_nrTbBaBtYgMplxUaktzCM&e=): Candidate skills extraction from resumes. But, dataset not available.

4.      Dataset preparation

a.      Looked at [<span class="underline">EDGAR</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__www.sec.gov_Archives_edgar_data_51143_000104746917001061_&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=z-pN3UACM1BRhaaXlQ_PMTv3ohbxku9KF5CM2-R9Urc&e=) data (html files)

b.      Looked at company annual reports (PDF format). Ex. [<span class="underline">PepsiCo</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__www.pepsico.com_docs_album_annual-2Dreports_pepsico-2Dinc-2D2019-2Dannual-2Dreport.pdf-3Fsfvrsn-3Dea470b5-5F2&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=C-GyfQrB1KfROZaI5L_kelOv0Z0nGtsYznYzyHbHSNI&e=)

 

 

**Papers (Literature Survey):**

1.      [<span class="underline">Dual Adversarial Neural Transfer for Low-Resource Named Entity Recognition</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__www.aclweb.org_anthology_P19-2D1336.pdf&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=MqFXvptYBe3ag4-1SKDxArMlIIm7eiQ9wvUZoyOBuO0&e=) (ACL'19)

·        **Idea:** adversarial modifications can help in knowledge transfer in low-resource settings

·        **Data Setting:** Cross-Lang (Eng to Spanish/Dutch), Cross-Domain (CoNLL English to Tweets English) knowledge transfer

·        **Model:** CNN-BiLSTM-CRF arch. with adversarial perturbations (introduced for robust training in low-resource setting)

·        GRAD loss (for shared parameters): basically linear transformation over self-attention then log likelihood of that.

o   Not sure how it would work without softmax? (1 - r) term: how does it work?

o   How is the linear transformation params decided?

·        Overall loss: label loss (source and target) (CRF) + GRAD loss

·        **Performance:** Comparable performance in high-resource and better performance in low-resource setting

 

2.      [<span class="underline">A Little Annotation does a Lot of Good: A Study in Bootstrapping Low-resource Named Entity Recognizers</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__arxiv.org_pdf_1908.08983.pdf&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=zx8wFmvZf-V-rABzUt1cbe-n3U2dkqxZOp8WIMFgvWU&e=) (EMNLP'19)

·        **Idea**: Transfer Learning then Bootstrapping (using manual active learning)

·        **Data Setting**: CoNLL English, Hindi, Spanish, Indonesian NER

·        **Model:** CNN-BiLSTM-CRF (for NER). Initially train Bilingual word embeddings then transfer knowledge to label unlabeled low-resource corpus. Train CNN-BiLSTM-CRF model for NER on it. Use marginal prob. to get entity spans with high uncertainty for human labeling. Augment to corpus and retrain NER model iteratively. Use partial-CRF for NER, since, sentences are partially gold annotated, for other parts of the sentence, the loss should not affect the model (since, those parts are not yet labelled)

·        **Performance**: +9 points in F1 (around) than other active learning or transfer learning techniques.

 

3.      [<span class="underline">CrossWeigh: Training Named Entity Tagger from Imperfect Annotations</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__arxiv.org_pdf_1909.01441.pdf&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=AT7JNUkHaxSssRZ1YZ470ex5WZcY25wQIrX8TnFFQ0s&e=) (EMNLP'19)

·        **Idea/Model**: Correction of training set errors using k-fold cross-validation setting where each sample is assigned a weight of being correct/incorrect

·        Bootstrapping

·        Is good where initial model has high confidence of being accurate (&gt;90% general accuracy)

·        **Data Setting**: CoNLL2003 English NER

·        **Performance**: Leads to ~0.5% improvements due to label corrections

 

4.      [<span class="underline">Improving Distantly-supervised Entity Typing with Compact Latent Space Clustering</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__arxiv.org_pdf_1904.06475.pdf&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=QlIQa5xJkmEspKcwknQ07cAeT8CdDLnuSvt0EmGxgT4&e=) (NAACL'19)

·        **Idea**: Distant supervision leads to noisy training data. Existing denoising methods rely on partial label loss objective which leads to confirmation bias (as it corrects data based on its own objective function). So, try to cluster similar entity mentions together instead.

·        **Model**: LSTM based for entity typing, minimizing KL divergence between predicted and actual entity labels

·        Compute embedding representation before passing to final classifier and make distance-graph out of them. Then, do label propagation on graph for clustering and revise embedding representation through a loss.

·        **Data Setting**: OntoNotes, BBN (news/wall street journal based)

·        **Performance:**  Compared to partial label loss, partial label embedding, joint representation of entity mentions and label types etc. approaches. 1-2% F1 improvement.

 

5.      [<span class="underline">Exploiting Structure in Representation of Named Entities using Active Learning</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__www.aclweb.org_anthology_C18-2D1058.pdf&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=X_ye6Pf3njtY9WobYHXRauEAT6Cvyb8UANq5qJoQ3u0&e=) (ACL'18)

·        **Idea**: Implicit structure in named entities (which is very task-dependent)

·        **Model**: Active Learning where user supplies some structures, model proposes more which the user can correct/reward. Edit distance among structures calculated. Rule generation based on pattern matches in corpus.

·        **Data Setting**: Person/Company mentions etc. from ACE

·        **Performance**: 3% F1 better on company, similar on Person (than CRF-based)

 

**Discussions:**

1.      **Model Design Goals**:

·        To design a simple and efficient architecture/model.

·        The approach should be general (across different types of entities), but clear focus for benchmark comparisons should be on a fixed set of entity types.

·        It should be robust (not overfit on benchmark data)

·        Alpha-numeric entites are different from word-based entities, because the meaning of the entity comes from the context around it.

·        Can think in the direction of pattern embeddings (pattern meaning/pattern inferencing) using local and global context, character representation may also help

·        For general word entities (person names, orgs etc.) comparison benchmarks exist

·        May also later want to extend the scope to identifying **latent entites** from text (like, religious views, political inclination)

·        CNN make be more efficient than LSTM-based models (achieving similar accuracies)

 

2.      Making sense out of privacy agreements (terms and conditions) and checking that they match the actual app accesses is also an active research topic (since, they are very much like legal texts)

 

3.      Some SEC filings like, management change forms (have person names), executives compensation forms (have person names), 10K, 10Q etc. can make up a publicly available dataset. Some other public datasets ([<span class="underline">OntoNotes</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__catalog.ldc.upenn.edu_LDC2013T19&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=-ZuhXqCWoPMYuhO5m9FEdB_QEWeNfUMAvsGnt5Dk4JY&e=), [<span class="underline">ACE</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__www.ldc.upenn.edu_collaborations_past-2Dprojects_ace&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=nHxI4F7IDN0pTPvvS_oh5nl-zRxtBYC-YUTCzrs0_vk&e=), [<span class="underline">CoNLL</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__www.clips.uantwerpen.be_conll2003_ner_&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=xvLNAExg3sEGYuw4Ey8z6t5jjVdPiRkFp6tpyAnsAC0&e=), [<span class="underline">NER Kaggle</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__www.kaggle.com_abhinavwalia95_entity-2Dannotated-2Dcorpus&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=peikXGZIPzqmKVg7NroFi4AsMtcq1sEM7TKg5AL378E&e=), [<span class="underline">PI and Medical</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__synthetichealth.github.io_synthea_&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=r6SPEBBAIh8VVWsFjK7_7r8eUp85FUtP0rVQxmBjFMA&e=), [<span class="underline">GENIA</span>](https://urldefense.proofpoint.com/v2/url?u=http-3A__www.geniaproject.org_genia-2Dcorpus_term-2Dcorpus&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=4nGD93ZnqvxIWVGb99SNtlkfoUQV1OkVE3x9VNZNW6U&e=))

 

 

**<span class="underline">2020/06/01 (Monday) 1-2pm CT:</span>**

 

**Our Proposed Model Architecture**:

·        Uses character-level information (or pattern-level information, since the data is labeled through regexes, so is sparse) for capturing the **intrinsic** meaning of a candidate named entity. Eg. For CD28, we could have \\w+ capturing CD, then \\d+ capturing 28. So, concatenation of 2 patterns, rather than character level which considers each letter separately.

·        It can use CNN for capturing **extrinsic** (**local context**) information that helps in named entity recognition. It can also try to use **global-level** (document-level) information for the task.

·        We can have 2 variants of the models. One which **jointly** captures the intrinsic and extrinsic properties.

·        Another one which has an **aggregator** module and first separately trains for intrinsic and extrinsic, then combines the info for the overall task. This might be easier to extend, debug and simpler from explanation perspective too.

 

**Dataset Discussions**:

1.      **CoNLL2003**:

·        Found from [<span class="underline">GitHub</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__github.com_juand-2Dr_entity-2Drecognition-2Ddatasets&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=dhtGyow_NGqsn5qLbHZnQOGXphZWLlPpJzJVqL35kYw&e=)

·        LM-LSTM-CRF framework: Total: **91**% (F1)

 

2.      **OntoNotes**:

·        Found and converted to CoNLL2003 format through [<span class="underline">GitHub</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__github.com_yuchenlin_OntoNotes-2D5.0-2DNER-2DBIO&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=pJgvNluxJ_Jf3hnjcHaN97vMrwIrcFko7P8mHyJz6QI&e=)

·        [<span class="underline">LM-LSTM-CRF</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__github.com_LiyuanLucasLiu_LM-2DLSTM-2DCRF&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=JkfjNag-L8gsRPaxJ1MFDT7wMDbUsluIAmzAd9pnUHg&e=) framework: Total: **84**% (F1) (after 27 epochs, one epoch training taking 1hr approx.)

·        Focusing on numerical entites:

o   **Date**: 84% (F1) \[Eg. 20th September, several years\]

o   **Time**: 65% (F1) \[Eg. 8:00, the afternoon of, an hour later\]

o   **Percent**: 91% (F1) \[Eg. Forty five percent\]

o   **Quantity**: 75% (F1) \[Eg. One thousand plus square kilometers\]

o   **Ordinal**: 75% (F1) \[Eg. First, last\] (low I-Ordinal F1 possibly because error samples in set, almost no training samples. Is very rare also, possibly, 'first and the last' time?)

o   **Cardinal**: 75% (F1) \[Eg. More than three hundred\]

o   **Money**: 90% (F1) \[Eg. Seven point one billion dollars\]

 

3.      In **our corpus**, generally in canonical form, IP addresses, currency etc. could be unigrams. But then there are varied ways of representation. So, there might be spaces in between making them bi-, tri-grams as well etc. So, we should think from the perspective of handling such cases as well.

 

4.      **GENIA**:

·        Has data in XML form (needs conversion to CoNLL2003 format)

·        Around 20k sentences

·        Protein molecule names seems to be alpha-numeric in most cases (so, we can test using that)

 

5.      **TM-VAR Corpus**:

·        Has a different format with named entity mentions and full text sentences.

·        DNA mutation labels seem to be alpha-numeric, but overall corpus size is small

 

6.      Some **other bio-medical corpuses** may help us prepare publicly-available alpha-numeric named entities datasets. A paper describing available bio-medical datasets and ways of converting it into CoNLL format: [<span class="underline">Paper</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__bmcbioinformatics.biomedcentral.com_articles_10.1186_s12859-2D017-2D1776-2D8&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=hHqbHflUL07ntfvCTOfSOH59uD4ta1NhCwhQ6BwOF34&e=) (BMC Bioinformatics, 2017), [<span class="underline">Dataset Information</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__pdfs.semanticscholar.org_f9cf_05a0b7dae45c06663f005fef5530943d28af.pdf&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=koxSBOA-skQ5R8PxJYAGB0tUfZRwjA93l6T1IOSqmvw&e=)

  

**<span class="underline">2020/06/08 (Monday) 2-3pm CT:</span>**

  

**Dataset Discussions:**

·        **GENIA**: Writing a parser code for interpreting the XML and possibly converting to CoNLL format for baseline models. It has nested entities for which we should ideally consider the lowest level (smallest) entity as the aggregated entity generally has some standard keywords after the main entity.

 

·        **PubMed(Partial)**: Also have access to a partially annotated PubMed (titles + abstracts) corpus which has Disease Names, Species, Genes etc. in JSON format.

 

**Model Architecture Discussions:**

·        [<span class="underline">FLAIR: Contextual String Embeddings for Sequence Labeling</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__www.aclweb.org_anthology_C18-2D1139.pdf&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=UlehKcJXaCMCOR0Zw1venaLofu8e9Pk8f966z7bLN4I&e=) (ACL'18): Contextualized string embeddings (character level modelling). Found that on standard NER corpus, adding character level information only slightly helps over word modelling. But this char-level representation may be helpful for our alpha-numeric entities.

 

·        [<span class="underline">Pooled Contextualized Embeddings for Named Entity Recognition</span>](https://urldefense.proofpoint.com/v2/url?u=http-3A__alanakbik.github.io_papers_naacl2019-5Fembeddings.pdf&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=-DFh6H_3PfAh5fzWZJd6m4csNVidyqIZcMP09LJBMXI&e=) (NAACL'19): For emerging entities (WNUT) as well, a variation of FLAIR embeddings gave a significant performance.

 

·        In our case, regexes are not explicitly available, we have some 50 or so samples of named entity matches. From these matches, we need to implicitly model intrinsic meaning (through implicit pattern/regex modelling) + extrinsic meaning (through local and global context) for NER task.

 

·        Instead of complete character modelling (like done in Deep Learning for Regex: blog or in character-embedding papers), we can try to standardize our examples into patterns and learn embeddings for those patterns, like:

 

609-123-4567

DDD-DDD-DDDD

D3-D3-D4

 

Foo S. Bar

U1L2 U1. U1L2

W W. W

A3 A1. A3

 

So, for each word: basic unit embeddings (pattern) + pretrained word embeddings (fasttext/glove) + subtype embeddings

Subtype embeddings (Ex.): email-address: person name + company name (person and company are the subtypes)

We could also think if we want to model our task as sequence labelling or span classification (like done in Deep Learning for Regex: blog)

 

 

**Pattern (Regex) Understanding with Deep Learning:**

1.      [<span class="underline">Deep Learning for Regex</span>](https://urldefense.proofpoint.com/v2/url?u=http-3A__dlacombejr.github.io_2016_11_13_deep-2Dlearning-2Dfor-2Dregex.html&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=H-NPkflsZbRYKSyZg-In2I4lZDUOLmqaQE_JBABsmgM&e=) (Blog): Learning to identify alpha-numeric Product ID from text sentences (deep char-level embeddings learning implicit regular expression formats) (data/code not available)

 

2.      [<span class="underline">Marrying Up Regular Expressions with Neural Networks: A Case Study for Spoken Language Understanding</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__www.aclweb.org_anthology_P18-2D1194_&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=4ZySZums9bAkmgnI3Okc-ujJOpDjLWxPBZ6sBvii_b4&e=) (ACL'18): Matching regexes in sentences and using the matched words for manipulating attention weights etc. in a standard BiLSTM framework for sentence classification (intent detection) and sequence labelling (slot filing) tasks. Dataset: ATIS + manually created regexes ([<span class="underline">data available here</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__pfllo.github.io_&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=pdXTl55LKjAtLfS45RGJ623Tqm5AYTsnITT15yXBotI&e=))

 

3.      [<span class="underline">DeepRegex: Generating Regular Expression from Natural Language Description</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__arxiv.org_pdf_1608.03000.pdf&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=YGMqf0jUheplx7aeK5lyO91CpNKGu_UlZ11YjRPujPU&e=) (EMNLP'16): Uses LSTM-based seq-2-seq arch., dataset(NX-RX, semi-manually created) has 10,000 regexes with single sentence descriptions. Code based on Pytorch and Lua ([<span class="underline">GitHub</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__github.com_nicholaslocascio_deep-2Dregex&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=LqtxhfPsDlYMECiGAQdfVDjXeZYU9TSBXHqt0Zj9FZU&e=)) (some positive and negative generated samples described [<span class="underline">here</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__news.ycombinator.com_item-3Fid-3D12269468&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=8XGxuHFIbh_plhGVQSa-Pzijgb9aY3kNXZzbKvUbZv0&e=))

 

4.      [<span class="underline">Understanding Regex using Deep Learning</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__medium.com_codist-2Dai_generating-2Dnatural-2Dlanguage-2Ddescription-2Dof-2Dregex-2Dusing-2Ddeep-2Dlearning-2D679248a95dab&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=YWDnjkl8NX6LhAKARYkQaKAHLH6lQzUXDLkslZ_DEJw&e=) (Blog): generating natural language description of regexes (opposite of DeepRegex). Used same data and model architecture. ([<span class="underline">data/code</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__colab.research.google.com_drive_1RTyUgR3hrUCfbbQb18TTTA0Wn9fCj0eW-23scrollTo-3DVMNuFaBwaZnc&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=nssvjLYHK-Puc1xequ20BwTPBFbEKt9qOHNWcCpCMao&e=))

 

5.      [<span class="underline">Generating Regular Expressions from Natural Language Specifications: Are We There Yet?</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__taoxie.cs.illinois.edu_publications_nl4se18-2Dregex.pdf&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=WzxPzsY0J66PVV_YxxG3Dyqvqa65Hb62puz8hAyNXC4&e=) (AAAI'18): Basically, evaluates DeepRegex (described above) on NX-RX dataset and a real-world RegexLib dataset (not available). Says, NX-RX dataset is synthetic and in real-world, this system, DeepRegex, still does not perform too well.

a.      Proposes that large real-world corpus should be made for better training

b.      Model should use positive and negative examples of regex match, apart from the seq-2-seq arch. for better regex interpretation

 

6.      [<span class="underline">Inference of Regular Expressions for Text Extraction from Examples</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__ieeexplore.ieee.org_stamp_stamp.jsp-3Farnumber-3D7374717&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=FAKSK_6aIpusgw2SJ3mbA1C1ZD4mRaz3h5TClSdP70I&e=) (TKDE'16): Automatically generating regular expressions from examples/given matches. (uses genetic algorithms) ([<span class="underline">slides</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__www.human-2Dcompetitive.org_sites_default_files_bartoli-2Ddelorenzo-2Dmedvet-2Dtarlao-2Dslides.pdf&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=zq2EjrB7XrjfYkOHZocj8LO9REhBIw9cuhocWTT4wzs&e=), [<span class="underline">code</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__github.com_MaLeLabTs_RegexGenerator&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=Yt3b8z7lMxdQrLiEvEtZgxEMTHuyeBbQIfvouNTNNGE&e=), [<span class="underline">demo</span>](https://urldefense.proofpoint.com/v2/url?u=http-3A__regex.inginf.units.it_&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=HL8IS_4H-K4YvD5ULScE5BXIelivW70L_VslgWuuSwQ&e=))

 

7.      [<span class="underline">Language coverage and generalization in RNN-based continuous sentence embeddings for interacting agents</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__arxiv.org_pdf_1911.02002.pdf&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=t5cS2wBplupfUG_9n_lEiQwQ1XUjErgB_bsqsi0PoqM&e=) (ArXiV'19): They take a CFG and all sentences generated by it, then feed these to RNN-based system to get sentence embeddings and check reconstruction/generation of any sentence in the CFG. They observe that RNN-based sentence representation is biased and does not generalize well. They propose a PCFG-based non-learning embedding for a given/known CFG, which can serve as a benchmark for comparing neural-net-based learnt sentence representations.

 

 

**Numeral Understanding with Deep Learning:**

1.      Numerical Polysemy (Challenge with numbers/alpha-numerics): Number '12' can have diverse meanings in different sentences based on context! It can be a month, date, age, size of plot etc. So, a single embedding for number 12 is not a good idea. Contextual embeddings would be better! Similar thing happens with words too, like, 'bank' but has less diverse range of meanings.

 

2.      [<span class="underline">Numeracy-600K: Learning Numeracy for Detecting Exaggerated Information in Market Comments</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__www.aclweb.org_anthology_P19-2D1635.pdf&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=-HmTnZ_07jI-1S30fFAGBBqr3zI34HDOrp_uCGwdH2Y&e=) (ACL'19): Prepared data from market comments on Reuters. Predict number range (among 8 classes) for filling a blank related to the market comment. Found Bi-GRU framework working best. ([<span class="underline">Dataset</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__github.com_aistairc_Numeracy-2D600K&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=q09Ln4FyKKT4YcWgxUZwXWqvwFlCAkL_HWdZDoUUfrQ&e=))

 

3.      [<span class="underline">Do NLP models know numbers?</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__www.aclweb.org_anthology_D19-2D1534.pdf&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=QV3HrP0h1V-EasXHlpuoYALA0wvY-TYkoFtk35nbvdY&e=) (ACL'19): Pretty good analysis showing that Word2Vec, GloVe, char-CNN embeddings, ELMo, BERT all have some good sense of number representation. Tested on numerical Q/A dataset ([<span class="underline">DROP</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__allennlp.org_drop&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=2g63MX1SMHWZzZWY1VrnReoUjKgly7iIkyA2bpWkWdY&e=) - made by AllenNLP). Found char-level embeddings are able to represent numbers better than word embeddings. CNNs arch. are able represent numbers better. ELMo gives better performance than BERT.

 

4.      [<span class="underline">Learning Numeral Embeddings</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__arxiv.org_pdf_2001.00003.pdf&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=Xlv4BA3EohJFTE2GjsRTQSC0wS_Cgx_OvD3Aeedaqbk&e=)(submitted to ICLR'20): Learning number embeddings by sampling some representative numbers out, embedding them, then for any other number, do soft clustering by learning a GMM over those representative number's embeddings.

 

**BioMedical:**

 

1.      [<span class="underline">Unsupervised Representation Learning of DNA Sequences</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__arxiv.org_pdf_1906.03087.pdf&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=LuB5kvFekHOfXq60EqV_XfWj3eyN0tb-H7s47QdgRkE&e=) (ICML-Comp. Bio'19): Use auto-encoder framework to embed long DNA sequences and the dense embedding representation from auto-encoder is used for evaluating on sequence classification tasks.

 

2.      [<span class="underline">Continuous Embeddings of DNA Sequencing Reads and Application to Metagenomics</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__www.liebertpub.com_doi_10.1089_cmb.2018.0174&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=Ns2JcjbeSJL78Ahw4CNmqDa5BxA5QNDj-Tu-LKGZxZM&e=) (Journal of Comp. Bio'19): Prepare FastText type char-level (capturing sub-word information) embeddings of DNA sequences.

 

3.      [<span class="underline">Learning Protein Embeddings for Machine Learning</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__pubmed.ncbi.nlm.nih.gov_29584811_&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=_iGSwt9Xztu-FDJVOqPdMImEQq-CKXNZIQRiJXZgZkI&e=) (Bioinformatics'18): Use Doc2Vec(gensim) considering amino-acid sequences (as docs) and individual fragments (as words). They use these embeddings for protein property prediction tasks. ([<span class="underline">code</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__github.com_fhalab_embeddings-5Freproduction_&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=TsyyDTmXd5Q3dZuBQoQY_2-JPvM1tdWJecM0j7toCrw&e=))

 

**Product/Entity Embeddings (Based on Word2Vec):**

1.      [<span class="underline">Pin2Vec: Related Pins at Pinterest: The Evolution of a Real-World Recommender System</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__arxiv.org_pdf_1702.07969.pdf&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=csr77mFblGeXt5H3Osw9G8BTIDKbY9UBzl6yzLzgPMY&e=) (WWW'17): Take related pins from browser histories of users as context for a given Pinterest pin topic and embed using skip-gram Word2Vec model. ([<span class="underline">blog</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__medium.com_the-2Dgraph_applying-2Ddeep-2Dlearning-2Dto-2Drelated-2Dpins-2Da6fee3c92f5e&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=Cep2dQ8Pj-PHbiyQvZxsxGsHT2Tv7qIbd0XFqY2BnLs&e=))

 

2.      [<span class="underline">Business2Vec: Identifying similar businesses using embeddings</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__medium.com_-40eniola.alese_business2vec-2Didentifying-2Dsimilar-2Dbusinesses-2Dusing-2Dembeddings-2Dpart-2Di-2D82962fd3ecac&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=5vID4tm36yOlXmHgrM1PitrVw6PEQ7A6ikZi_6zzFF8&e=) (Blog): Use Gensim's Doc2Vec for preparing paragraph embeddings of business descriptions to find similar businesses.

 

3.      [<span class="underline">Item2Vec: Neural Item Embedding for Collaborative Filtering</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__arxiv.org_ftp_arxiv_papers_1603_1603.04259.pdf&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=glua5tqrYITcIXx96Ky_ZsqD7oGo6H1HyNg-avTZw74&e=) (MLSP'16): Use Word2Vec skip-gram model over item names with neighboring context being other items which the user bought together with this product (in a shopping basket)

 

**General:**

1.      [<span class="underline">NCRFpp</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__github.com_jiesutd_NCRFpp&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=0yvCXeZ_2CHumtexMh9QttHZ9GT4i7fcRv8a5QG4aPM&e=): Popular codebase for CNN-LSTM-CRF framework:

 

2.      [<span class="underline">A Survey of Word Embeddings for Clinical Text</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__www.sciencedirect.com_science_article_pii_S2590177X19300563&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=UBocHjbSaPBnFXgoAAH5ntqXQLIjX-poGVAjcRf7QAo&e=) (Biomedical Informatics, Dec'19): Talks of several datasets like MIMIC-III etc. and techniques like ELMo, BERT, Word2Vec etc. used with their variations for better modelling of rare clinical terms.

 

3.      Entity/Product Names are also many times alpha-numeric (like, 'Samsung S10'): Preparing product name embeddings could also be considered similar to our task of having a good vectorized representation for our alpha-numeric entites.

 

4.      [<span class="underline">Word Embeddings for Fuzzy Matching of Organization Names</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__www.rosette.com_blog_word-2Dembeddings-2Dfor-2Dfuzzy-2Dmatching-2Dof-2Dorganization-2Dnames_&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=IieFjWez45pe-5v7wjZkn3ZTmg902wf8nyCNR-YCvGc&e=) (Blog): Use phonetic and textual similarity features for word similarity on organization names (Eagle Pharma Inc, Eagle Drugs Inc.) Model arch. etc.details not mentioned.

 

**Others (not too related):**

1.      [<span class="underline">Weakly-supervised Relation Extraction by Pattern-Enhanced Embedding Learning</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__arxiv.org_pdf_1711.03226.pdf&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=Z99BJ0D8-BWeo6WXOT1rCMiXkyD2XDz2Y2z_FqcjiYA&e=) (WWW'18): Given (X, Y) entites and pattern X is capital of Y. Clustering similar phrasal patterns based on similarity heuristics for relation extraction between a new pair (A, B) based on phrases where they co-occur.

 

2.      [<span class="underline">Symmetric Pattern Based Word Embeddings</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__www.aclweb.org_anthology_K15-2D1026.pdf&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=TUtQZfsa07Dw_MArPjqVcVO7o5xwTGJrcby1ZzYL-OM&e=) (ACL'15): Makes sparse one-hot correlation matrix of words with their similar words (identified using patterns (X and Y etc.), then PMI (mutual information) based training objective.

 

3.      [<span class="underline">Natural Alpha Embeddings</span>](https://urldefense.proofpoint.com/v2/url?u=https-3A__arxiv.org_pdf_1912.02280.pdf&d=DwMFAg&c=jf_iaSHvJObTbx-siA1ZOg&r=qSURBpJm40z2tJQpQ2CSkLUO3jSNJW_r6mReWAZ3ymg&m=7PGqQGWh1hEKpt8wh1ImpGJ7l6iwgVk8Zy5sjsJRx2E&s=xtCW8yjbrXHcVXCNeJbKVg6tK2ccqrB4ex-H2ErXIVY&e=) (Dec'19): Some statistical tweaks / formulations for better modeling of item embeddings.

**<span class="underline">2020/06/15 (Monday) 1-2pm CT:</span>**

BOX - for raw data

Can keep processed data in GitHub repo

GPU on Server: ssh <jatin@flowers.sl.cloud9.ibm.com>

**Datasets:**

MIMIC-III Dataset

-   Last year worked on MIMIC-III dataset which has clinical record and patient reports, but the actual names, addresses, phone numbers are removed from the dataset and replaced by masked special tokens like \[NAME\], \[PHONE\] etc.

-   We may try to artificially generate the data and fill in (for intrinsic mention information capturing)

-   The original data can still provide us context training (out-of-the-box)

-   But mostly the entity types are general ones, not SSN, Credit Card Num, Driving License etc. (which are alpha-numeric and are our main focus).

Synthea Corpus

-   Has personal information, SSN, medical condition etc. but without context and in JSON form.

-   This data is synthetically generated

-   The SSNs have format 999-…., Driving License has S999…, Names has numbers within it so that it becomes artificial (like, John819). SSNs do not start with 999-… So, this artificial data does not help the model learn the validity of SSN etc., intrinsic properties of the client personal entities may not be captured well by this data as well (since, it is not real)

Some other data is also available with info including UPS tracking number, Passwords etc.

**Models / Approaches:**

-   WordLSTM, WordCNN, CharLSTM, CharCNN - models are compared for NER tasks in terms of performance and efficiency (in some recent research work). Found that LSTM is slightly better than CNN in performance, but CNN is much more efficient than LSTM

**Sequence Labelling Approach:**

-   We may develop entity-mention (intrinsic) model and a separate context-capturing model and then co-train them!

-   We may also ensemble the 2 above models

-   Another option is a single model which captures both things jointly

-   Have to decide on the right approach based on simplicity of the architecture and performance gains

**Span Classification:**

-   We may view this task as a Entity-Boundary detection and then Span Classification problem. Initially for entity boundary detection, we may use existing general approaches.

-   First we do noun phrase chunking using [SciSpacy](https://allenai.github.io/scispacy/) etc. and then train model to do multi-class classification for each chunk into one of the output entity types. (Can use both intrinsic info and external context for span/chunk classification task)

-   Since many entities may not be noun phrases, we may train a separate model that just tries to identify spans (of interest). Could be any span. Since, phone number spans are not noun phrases etc. But this entity boundary detector may have to cover a very diverse range of entities - so may be challenging to train a single model. We may train a boundary detector for each entity type.

-   Span classification task has much lesser params to train, than sequence labelling (so, may be feasible for low-resource NER task)

-   Although, Sequence Labelling has been found to be very effective at this (and possibly better than span classification). But that’s in general, may not be true on our client-info dataset in low-resource setting.

-   For start and end boundary detection, we may train separate detectors (2 index vectors), like done in DeepLearning for Regex blog. So for each entity boundary, we get 2 vectors to train. In low resource setting, we may train only one vector for each entity type, since, working with at least 10-20 different entity types, we may need more data otherwise.

**Character / Pattern Modeling:**

-   Char is in general better (more informative) but may require more params to train. Char embeddings may be sparsely trained but may learn valid range values etc. implicitly. Pattern embeddings may train better but will not be able to learn range of valid values for an entity (using the encoding techniques discussed next)

-   We may normalize mentions like, John-DNA.987 may become Ulll-UUU.ddd (upper case, lower case, digit handling), I-U.ddd (initial letter capital, all lower case word encoding etc.) This will significantly help reduce no. of params to train!

-   But the model will not be able to learn valid ranges (like SSN does not start with 999 - if we go by this approach)

-   Notice that no. of digits is kept intact in 2nd pattern encoding technique, alphabetic words are converted to only 1 letter (U - all upper, L - all lower, C - camel case, I - Initial letter caps, M - mixed). Punctuations are retained as it is, in this model.

-   We may also concatenate pattern and char representations if we find that patterns are capturing non-intuitive things which may not be easily captured using chars.

**Bootstrapping Approaches (for Training):**

-   In our case, we can’t really do manual annotations

-   Snowball bootstrapping techniques can be used

-   So, automated bootstrapping techniques can be helpful, based on some weighting scheme/heuristics, so that we add high confidence samples into the training dataset at each step. But the problem of Snowball drifting may occur which needs to be properly handled. There might be errors introduced in the training data which may magnify with more and more steps of bootstrapping.

-   Earlier worked on a paper utilizing semi-supervised active learning (2012) which was clustering based, without human annotations.

-   Cluster automatically identifies the best candidates at each step and the clustering technique is overall classifier independent. So, its a general bootstrapping framework that can be tried by multiple different types of classifiers.

-   One of our major goals is to come up with some **automated way of judging model outputs** and then do **bootstrapping**.

**<span class="underline">2020/06/22 (Monday) 1-2pm CT:</span>**

**Models / Approaches:**

-   Nester Bio-Medical NER (Penner) etc. approaches may be extrapolated for our client personal alpha-numeric low-resource NER task.

**Feeding in Type information to the model (and possibly bootstrapping)**

-   SSN occurs close to person name. Eg. John’s SSN is 123-4567-890. So, \[PERSON\]’s SSN is 123-4567-890. This PERSON tag information can help model identify the SSN.

-   One-hot embeddings of tags (gold tags) for each token in sentence can be fed as additional input to the model. While predicting output for a token, the model should somehow not consider the tag information for the current token

-   But during query time, these gold tags would not be available. **So, need to think about the testing flow**.

-   Also we could utilize POS, Dependency Parse features since, SSN does not occur as **subject**, while PERSON name generally occurs as subject.

-   SciSpacy could be used for clinical NER tags being fed as input to model

<!-- -->

-   We could initially feed O tags for each token and train to get model’s output classification tags. These tags are then fed as input in next step to get better tag predictions from the model. Classic bootstrapping. But we need to take care of snowball drifting. In the testing stage also, we initially feed in with O tags and run multiple iterations possibly to reach output labels.

**Chunking inputs to model:**

-   CNN/LSTM take fixed size inputs. To enforce that, we add &lt;PAD&gt; tokens/chars for short sentences/words, truncate larger sentences/words.

-   Instead of that, we could fix sentence length to say, 25. Then, divide sentence into 25 chunks. Each chunk can have more than one token. We take individual token representation then do max-pooling/avg-pooling to get chunk-level representation.

-   Major advantage is, we don’t truncate. Hence, we do not loose information. In NLP/Linguistics, every small part of sentence is very important in deciding the semantics of the sentence.

-   In this approach, we may have spaces within the token/chunk representation. Special weightage could be given to the space char, or we may try to capture length of individual tokens separately in the overall representation. (Eg. “ABC defg” -&gt; “U3SL4” is the pattern representation for this chunk with 2 tokens. Through the pattern we know that there are 2 tokens, first in all caps and second in all lower, their lengths. We can also model the space char in between.)

-   Or we could just concatenate the overall chunk embedding with char representation of no. of tokens in that chunk.

**<span class="underline">2020/06/24 (Thursday) 1-2pm CT:</span>**

**Models / Approaches:**

**Capturing Context:**

-   Currently use GloVe word embeddings concatenated with char-level representation. Since the word-embeddings are pre-trained they capture the general context from general corpus, but not the localized context of the current sample.

-   So, using max/avg pooling or concatenation in a context window of Size=5 both sides (left and right) could help.

-   Passport Num generally has country name in context, telephone number generally has person name in context.

-   SSN tends to occur with BankID, credit card number in context.

-   The context could be entire **sentence** (taking max. works better as per some prep. papers) or a **sliding window**

**Bootstrapping / Masked LM Perspective:**

-   We could think of masked language model perspective. The current token’s entity type is masked and predicted using the one-hot encoded entity type information of context tokens

-   For entity type classification, in a bootstrapping approach, we could take the best label for a token and feed it as input in the next iteration of the model. Or, we could think from a soft-classification perspective where we feed all tags for a token back, but with weights (or probability scores).

-   But this soft classification could introduce a label-bias. Some misclassification could be given high weightage and propagate? Would that not occur in the hard classification task?

**Next Steps**:

-   Remove O tags from accuracy calculation (done)

-   Do qualitative analysis of results (done)

-   Check distribution of O tags in OntoNotes/CoNLL/Genia corpuses.

**<span class="underline">2020/06/28 (Monday) 1-2pm CT:</span>**

-   Discussed results of Char-CNN, Word-LSTM, LSTM+CRF arch.

-   Implemented and included basic pattern level representation concatenated with char embeddings

**Next Steps:**

-   **Ablations:**

    -   Char

    -   Pattern

    -   Word

    -   Char + Pattern

    -   Pattern + Word

    -   Char + Word

    -   Char + Pattern + Word

<!-- -->

-   Try multiple variants of Patterns (including condensed representation)

-   Measure accuracy with/without O tags

-   Capture sentence context or sliding window context with CNN arch.

-   CNN + CRF arch implement to check if B-Tag, I-Tag classifications are improved?

-   Check word length, sentence length stats for fixing the input dimensions to model

-   Can do 10%, 20%, 30% data training on GENIA Corpus and plot to get low-resource setting performance results

**<span class="underline">2020/07/02 (Thursday) 1-2pm CT:</span>**

**Discussion of Results related to:**

-   GENIA Corpus stats (word length, sentence length)

-   Ablation study

-   TMVar corpus size

-   DNA Mutations / Protein Mutations modeling by special tokenization rules and hand-crafted features based CRF (Eg. c.2708\_2711delTTAC -&gt; \[c\]\[.\]\[2708\]\[\_\]\[2711\]\[del\]\[TTAC\] {7 tokens})

-   Model settings (like, loss function used, learning rate, epochs training, pre/post padding done etc.)

**Next Tasks:**

-   CRF combine with CNN

-   Other Pattern based embeddings implement

**<span class="underline">2020/07/06 (Monday) 1-2pm CT:</span>**

Discussions on qualitative study

-   Tag pair ambiguity/similarity (For each pair of gold tag1 token which got predicted as gold tag2 token, we maintain the count)

-   Token diversity for a tag (how many unique tokens are there for a gold tag).

-   Tag diversity for a token (how many gold/predicted tags associated with a token in the corpus). For predicted tag, for each token, we have a single output tag.

Discussion of results of simple CNN arch. on OntoNotes, TmVar, CoNLL2003, Genia corpuses.

In our real dataset of client personal information with National IDs etc., we may expect to have lots of 9-digit numbers which are not National IDs. So, there will be a lot of O tags in the data. Task is to reduce the no. of false positives (to reduce false alerts), without increasing the false negatives. Currently we work with documents of only selective countries. If our system has to work with more countries, then the diversity of National ID formats will also be more.

**Models / Approaches:**

We could think of training to find National IDs based on Person Names in the context (from Person Name, we could try to inference the country and then use that information for National ID format) (for 5-10 different countries, this could be trained even without context, just based on pattern level information itself)

**Span Classification Perspective:**

Ex. May have to deal with phrases like, “a very smart John” and then do some noise removal (post-processing) to get the person name, “John”. Could also think of this from B/I tags perspective.

**<span class="underline">2020/07/09 (Thursday) 1-2pm CT:</span>**

Wrote script to check which all tokens are getting misclassified for each tag, so that manually looking at them, we may be able to identify some error pattern which can then be fed to the model for training. But could not find any specific pattern.

Also found that all tokens which have only 1 single tag associated with them in the dataset are generally correctly classified. The tokens that have multiple tags are classified into one of the tags. So for the other tags these tokens get reported as misclassifications. To correct this issue, the plausible way seems to include contextual information.

Discussions of results on GENIA Corpus for

-   Different pattern embeddings

-   Upper/Lower case char embeddings

-   Digit pattern (d) or retaining digits (0-9) in the pattern

-   CNN + CRF

-   CNN + LSTM

-   CNN + LSTM + CRF

Results of the above models and LM-LSTM-CRF model on GENIA Corpus, OntoNotes corpus also discussed.

**Next Tasks:**

-   Summarization of all results

-   Code organization

-   Entity Type information fed into the model (**get more stable results**)

-   Based on the new types of features/information being input to the model, can we think of some more intuitive architecture/loss function etc.

-   For separate models for capturing context and pattern information, could we think of some co-training objective (also look into multi-view learning)

**<span class="underline">2020/07/13 (Monday) 1-2pm CT:</span>**

**Discussion of results on GENIA Corpus:**

-   Char (Upper + Lower) is similar to Char (Lower). This means, in this dataset, for various entities, uppercase vs. lowercase does not change the meaning of token.

-   Pattern Analysis:

    -   1. Pattern (Condensed): “ABC-deFG-123” -&gt; “u-m-ddd”

    -   2. Pattern (Condensed) + Digits: “ABC-deFG-123” -&gt; “u-m-123”

    -   3. Pattern: “ABC-deFG-123” -&gt; “uuu-lluu-ddd”

    -   4. Pattern (Condensed) + Digits + WordLen: “ABC-deFG-123” -&gt; “u3-m4-123”

-   Performance: (1) &lt; (2) &lt; (3) &lt; (4). 3 and 4 capture the sub-word length information which may be helpful. 4 is a compressed representation and still gives the best results.

-   CNN + LSTM + CRF model (with Word + Char + Pattern embeddings) gives the best results as compared to all its ablations.

-   The CRF takes more much more time to train as compared to just the LSTM. (Approx 7 times more time)

**Type Information Results:**

-   Directly feeding in the type info of current token will lead to model cheating and copying input to output.

-   Taking context window (in which we make those entities hot which are present in a sliding context window of size=5 on either side), we observe that if LSTM used, then again model cheats based on surroundings.

-   Without LSTM, simple CNN, CNN + CRF give comparable results. Although overall simple CNN gives slightly better results as of now (partial training). It is however expected that at token level with B/I-tags, CRF should give better results than simple CNN.

-   Simple CNN with type(from context window) gives much better results (+10% F1 approx) than just CNN without type info.

**Next Tasks:**

**Type Information Experiment:**

(Say we have 4 entity types in dataset)

Protein

Gene

DNA-Sequence

DNA-Type

(1) Split the entity types: (to mimic real life problem, target entities should have less labelled data available than the training entities)

**&lt;target&gt;**

DNA-Sequence

Protein     

**&lt;training&gt;**

DNA-Type

Gene

(2) Use same architecture – CNN-LSTM-CRF (may remove CRF if it takes too much time to train)

**Model1**: use only current token level information (Char + Word + Pattern\[condensed+word length+digits\]), predict the target types Y={target types}

**Model2**: Model 1 input + type information in context {training types} 4-dimensional one-hot vector (in current example, including the B/I-Tags), predict the target types Y={target types}

Any performance gains in Model2 will help us understand the contribution of type information towards it.

**Models / Approaches: **

ACL20 Paper/Code: <https://github.com/ShannonAI/mrc-for-flat-nested-ner>

-   Models NER as MRC task (uses BERT)

-   Main idea is, sequence labeling takes entity tags as one-hot embeddings and does not understand the implicit meaning of entity classes.

-   Converts each entity into a question/query (taken from Annotator Guidelines), concatenates with actual sentence input and BERT models outputs weights.

-   Uses those weights to train start-index, end-index, span-matching classifiers (total 3 losses optimized)

-   Using a single classifier for both start and end would give only one span match for an entity in sentence, which is overcome by training separate start and end classifiers.

-   Achieves SOTA results on most NER datasets for both flat-NER(CoNLL, OntoNotes) and nested-NER(like GENIA Corpus) (GENIA: +5% F1)

-   Also shows that with less training data, gives better performance (so, good for low-resource NER)

-   Shows zero-shot transfer effective from CoNLL03 training to OntoNotes testing. (But as per my understanding, even during testing time, it feeds in the annotator guidelines to the BERT model as input)

-   GENIA: Benchmark: 83.75% F1 (<https://paperswithcode.com/sota/named-entity-recognition-on-genia>)

**Our Methodology:**

-   We can borrow this strategy of better entity meaning modeling as input to our model (either modeled as span classification or sequence labeling setup). Entity embeddings or synonyms or examples can be encoded into some meaningful representation.

-   We already have pattern information as input

-   We are also feeding in the surrounding type context as input, which is a new kind of input and can be helpful (expected from the fact that telephone numbers occur close to addresses, SSN close to credit card details etc.)

-   Also need to read through other related works to understand their model architectures etc.

**<span class="underline">2020/07/16 (Thursday) 1-2pm CT:</span>**

-   **Tag segregation method** (based on frequency). Took top 6 tags as input(training) tags and others remain in the target.

<!-- -->

-   **Results discussion**: Tag information is found to be helpful. Dense tag embeddings work better than one-hot tag representations. We see quite some difference between overall Macro and Micro F1 scores. This could be due to tag distribution bias in the dataset because of which some classes may be performing much lesser than the rest of them, causing the difference in the values)

<!-- -->

-   **Word2Vec tag embedding preparation**: Replace entity mentions with corresponding entity tags and train 50-dim embeddings (11k sentence, 9k vocab size, 10 iterations over the entire dataset using skip-gram word2vec). For differentiating between B-/I- tag variants of a Tag, we add one additional dimension to the tag representation. To special handle the O(Other) tag and &lt;PAD&gt; tags again, we may make them zero-vectors and have a special bit specifying whether its O(other tag) or &lt;PAD&gt; (out of sentence tag).

<!-- -->

-   **TSNE projections**: some tags like protein-substrate, protein-molecule etc. are very close to each other in the vector space. Tag embeddings also share proximity with mention embeddings and related terms (like, DNA-domain-or\_region found close to word, “mutation” etc.)

<!-- -->

-   **New model techniques**: Instead of treating multi-class classification problem with one-hot independent classes, prepare dense class embeddings and model’s objective is to align mention representation close to category’s (or tag’s) embedding representation. (Cosine + KL Div / Cross entropy) (gold cosine distribution can be made by inter-tag cosine scores from pre-trained tag embeddings)

<!-- -->

-   Even in our **low-resource settings**, we can have small no. of gold labels, some regexes using which we annotate large amount of unlabeled data with noisy labels. Using these labelled data, we train Word2Vec model to prepare tag embeddings capturing context. Also, we may use char-embeddings (like, FastText, Flair) to capture the tag word’s semantics (to distinguish between tag, ‘Korean National ID’ and ‘Indian National ID’)

<!-- -->

-   We may also use the cosine similarity concept in **automated evaluation** during **bootstrapping**. Gold (manually labeled) entity mentions have some alignment with entity tag embeddings. Similarly, noisy labelled training data mentions too. This can help in classifying the new noisy labels as correct or not etc.

<!-- -->

-   **Negative cosine scores** may need handling. Either use absolute value, since the ultimate aim is that model should try to report cosine distance as close to 0 as possible. (Means entity mention embeddings are fully aligned with entity tag/category embedding representation). Or, may try using max-margin loss etc. (like used in [CosineEmbeddingLoss](https://pytorch.org/docs/master/generated/torch.nn.CosineEmbeddingLoss.html) - PyTorch) Also, NegativeLogLikelihood requires input to be probability distributions (which is not true in our case. Our’s is cosine values distribution, hence, may try using KL-Divergence instead)

<!-- -->

-   To handle **label bias** due to large no. of O (Other) tags in the dataset, we may take inspiration from [Focal Loss](https://github.com/clcarwin/focal_loss_pytorch) objective (used for background labels in object detection task).

**Next Tasks:**

-   Model architecture implementation

-   Check SOTA GENIA results (and fine-grained, if available), or else we may have to rerun their system to replicate their results. Also check if reported F1 score is Macro/Micro F1.


**<span class="underline">2020/07/17 (Friday) 4:30-5:30pm CT:</span>**

**Observations Discussed:**

-   Cosine similarity based strategy using pre-trained tag embeddings and using MSE loss implemented

-   The loss converges but the model develops extreme bias towards 1 or 2 labels (B-G\#DNA\_N/A or I-G\#mono\_cell). All the tokens (including the O tag tokens) get classified into this tag and the model attains a sub-optimal local minima

-   Also with increased epochs, the model fluctuates among these two above mentioned tags, but does not try to evenly distribute its results across multiple tag categories. It just goes all in one direction.

-   This could be because the the loss function currently is losing out on all the independence among the tag categories.

-   It is also possible that these 2 tags have their tag representation very similar to the O tag, so cosine distance with the O tag is getting minimized through these.

**Next Steps:**

-   Check the cosine similarity of various tag embeddings learnt using word2vec and identify the 3 tags most similar to each tag

**Model / Approaches:**

-   The gold label cosine distribution to be 1-hot with only cosine similarity value as 1 for only the right target tag and 0 for all the other tags. Then convert the model’s output cosine distribution into log probability distribution and use cross-e$

-   Instead of training a multi-class classification objective with softmax, use sigmoid function for each target class dimension and train T binary classifiers (jointly)

-   Use the MRC-BERT paper’s code and instead of feeding in annotation guidelines, feed tag name as query context to BERT model and check results

-   On CoNLL03 corpus, with the cosine objective, do we get better results? If yes, then the problem could be due to less accurate tag embeddings being formed in the GENIA Corpus (and we can focus on improving that), else, its an issue of the new loss$

