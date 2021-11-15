from utils import *

import re

dirname = 'corpora/TREC'

ids = {
	'DESC': 0,
	'ENTY': 1,
	'ABBR': 2,
	'HUM' : 3,
	'LOC' : 4,
	'NUM' : 5,
}

def clean(line):
	ss   = line.split(' ')
	tag  = ss[0].split(':')[0]
	text = ' '.join(ss[1 :])
	text = tidy(text)
	text = re.sub('o \'neal', 'o\'neal', text)
	text = split_symbols(text)
	line = f'{ids[tag]}\t{text}'
	return line

trains = [clean(line) for line in read_lines(f'{dirname}/train_5500.label', 'Windows-1252')]
tests  = [clean(line) for line in read_lines(f'{dirname}/TREC_10.label',    'Windows-1252')]

write_lines('data/trec_train.txt', trains)
write_lines('data/trec_test.txt',  tests)
