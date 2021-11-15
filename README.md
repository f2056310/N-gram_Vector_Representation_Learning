# Learning Word Ngram Representations Based on Relationships Between Ngrams of Different Lengths

This repository contains the implementations of our paper,
the C++ source for training ngram vectors and the python script for data augmentation using the ngram vectors.
These implementations can reproduce the main result.

You shall obey the license.
We make the ownership of the resources clarified after the paper is accepted and officially published.

## How to train ngram vectors

### Building

To build the executable ndl2vec, run as

```shell
make
```

A POSIX system and a C++11 compatible compiler are required.

### Training

To train ngram vectors, run ndl2vec as

```shell
./ndl2vec input.txt vectors.ndl
```

The input file needs to be composed of sentences represented as space-separated word sequences.
The ngram vector file is compatible to that of word2vec.

Please modify the source itself and rebuild it if you want to change hyper-parameters.

## How to evaluate data augmentation using the ngram vectors

### Setup

Install package as

```shell
pip install -r requirements.txt
```

### Training

To train a model, run train.py as

```shell
python train.py --ndl enwiki.ndl --input train.txt --model model.h5 --aug 10
```

The input file needs to be represented as

```csv
label1 [tab] words1 [newline]
label2 [tab] words2 [newline]
...
```

Words and punctuation marks need to be separated by spaces.

### Evaluation

To evaluate the trained model, run eval.py as

```shell
python eval.py --ndl enwiki.ndl --input test.txt --model model.h5 --aug 10
```

The input file needs to be represented as

```csv
label1 [tab] words1 [newline]
label2 [tab] words2 [newline]
...
```
