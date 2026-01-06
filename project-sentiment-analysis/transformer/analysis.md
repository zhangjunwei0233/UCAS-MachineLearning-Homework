# Report

## Data cleaning and preprocessing

### Dataset features

label distribution uneven but reasonable:

- label 0: 7072
- label 1: 27273
- label 2: 79582
- label 3: 32927
- label 4: 9206

### Dataset cleaning

Datasets already clean.

### Data preprocessing

1. split training set to 10% validation set and 90% training set.

2. renamed columns

Original dataset entry:

```tsv
PhraseId	SentenceId	Phrase	                                                                        Sentiment
2	        1	        A series of escapades demonstrating the adage that what is good for the goose	2
```

Preprocessed dataset entry:

```tsv
PhraseId	SentenceId	text	                                                                        label
2	        1	        A series of escapades demonstrating the adage that what is good for the goose	2
```

## Model selection

![image-20251225122023570](C:\Users\m1761\AppData\Roaming\Typora\typora-user-images\image-20251225122023570.png)



Sentiment analysis is a typical task inside text classification domain. For transformers, it is reasonble to use encoder only models like bert, because the input is a sequence of words while the output is only one label of classification result.

I select three encoder only models:

1. distil-bert-base-cased: classical pretrained encoder-only model

2. distilbert-base-uncased-finetuned-sst-2-english: classical sentiment analysis model finetuned from distilbert

3. [twitter-roberta-base-sentiment-latest](https://arxiv.org/abs/2202.03829): twitter's sentiment analysis model

**All models are tuned on full parameter!** (tested tunning only on Model head, but performance is NOT ideal)

## Training and choosing the right hyperparameter

basic parameters:

- epochs: 2
- optimizer: AdamW
- lr_scheduler: cosine
- warmup_ratio: 0.1
- weight_decay: 0.01


I tested different **learning rates, batch size** on these models and chose the best. Example with model distilbert-base-uncased-finetuned-sst-2-english (on validation set accuracy):

|Learning rate\Batch size|16|32|64|
|:--:|:--:|:--:|:--:|
|1e-5|**71.80**|71.45|71.20|
|2e-5|71.39|71.51|71.44|
|3e-5|71.20|71.27|71.47|

## Results



Final test set accuracy for each model:

1. distil-bert-base-cased: 0.69195
2. distilbert-base-uncased-finetuned-sst-2-english: 0.68902
3. twitter-roberta-base-sentiment-latest: 0.70378



![image-20251225121750699](C:\Users\m1761\AppData\Roaming\Typora\typora-user-images\image-20251225121750699.png)



## Critical notes

This report contains One Extra bonus point: the twitter-roberta-base-sentiment-latest model is a recent model (2022), the paper is already linked in the Model Selection seciton above.