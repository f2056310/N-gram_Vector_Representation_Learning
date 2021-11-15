from utils import *

import os
import re

dirnames = [
	'corpora/CustomerReviewData',
	'corpora/Reviews-9-products',
	'corpora/CustomerReviews3',
]

def clean(line):
	ss = line.split('##')
	if len(ss) != 2:
		return None
	review, text = ss
	text = re.sub('-LRB-', '(', text)
	text = re.sub('-RRB-', ')', text)
	text = re.sub('<p>', '', text)
	text = re.sub('-- >', '', text)
	text = tidy(text)
	text = split_symbols(text)
	
	p = '[+' in review
	n = '[-' in review
	if p and not n:
		return f'1\t{text}'
	if not p and n:
		return f'0\t{text}'
	return None

trains, tests = [], []
for dirname in dirnames:
	for filename in os.listdir(dirname):
		if filename.lower() == 'readme.txt' or filename[-4] == '.xml':
			continue
		lines = []
		for line in read_lines(f'{dirname}/{filename}', encoding='Windows-1252'):
			line = clean(line)
			if line is not None:
				lines.append(line)
		for i, line in enumerate(lines):
			(trains if i < len(lines) * 0.9 else tests).append(line)

write_lines('data/cr_train.txt', trains)
write_lines('data/cr_test.txt',  tests)
