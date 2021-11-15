from utils import *

import re

dirname = 'corpora/stanfordSentimentTreebank'

text_lines  = read_lines(f'{dirname}/datasetSentences.txt', 'Windows-1252')
label_lines = read_lines(f'{dirname}/sentiment_labels.txt', 'Windows-1252')
split_lines = read_lines(f'{dirname}/datasetSplit.txt',     'Windows-1252')
dict_lines  = read_lines(f'{dirname}/dictionary.txt',       'Windows-1252')

def to_binary(score):
	if 0.0 <= score <= 0.4: # --, -
		return 0
	if 0.6 <  score <= 1.0: # ++, +
		return 1
	return None

id2labels = {}
for line in label_lines[1 :]:
	id, score = line.split('|')
	label = to_binary(float(score))
	if label is not None:
		id2labels[id] = label

text2labels = {}
for line in dict_lines:
	text, id = line.split('|')
	if id not in id2labels:
		continue
	text = re.sub('-LRB-', '(', text)
	text = re.sub('-RRB-', ')', text)
	text2labels[text] = id2labels[id]

id2splits = {}
for line in split_lines[1 :]:
	id, num = line.split(',')
	id2splits[id] = int(num)

trains, tests = [], []
for line in text_lines[1 :]:
	id, text = line.split('\t')
	text = re.sub('-LRB-', '(', text)
	text = re.sub('-RRB-', ')', text)
	if text not in text2labels:
		continue
	label = text2labels[text]
	split = id2splits[id]
	text  = tidy(text)
	text  = split_symbols(text)
	(trains if split in (1, 3) else tests).append(f'{label}\t{text}')

write_lines('data/sst2_train.txt', trains)
write_lines('data/sst2_test.txt',  tests)
