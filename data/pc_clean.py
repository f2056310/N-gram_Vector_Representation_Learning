from utils import *

import re
import xml.etree.ElementTree as ET

def clean(line):
	line = re.sub('<p>', '', line)
	line = re.sub('AT&T', 'AT&amp;T', line)
	line = re.sub('&#1[30];', '', line)
	line = re.sub('& ', '&amp; ', line)
	try:
		elem = ET.fromstring(line)
	except:
		line = re.sub('&', '&amp;', line)
		elem = ET.fromstring(line)
	if elem.text is None:
		return ''
	text = tidy(elem.text)
	text = re.sub('</?[bi]>', '', text)
	text = split_symbols(text)
	return text

dirname = 'corpora/pros_cons'

pros = read_lines(f'{dirname}/IntegratedPros.txt')
cons = read_lines(f'{dirname}/IntegratedCons.txt')

trains, tests = [], []
for i, text in enumerate(pros):
	text = clean(text)
	if len(text) < 8:
		continue
	(trains if i < len(pros) * 0.9 else tests).append(f'1\t{text}')

for i, text in enumerate(cons):
	text = clean(text)
	if len(text) < 8:
		continue
	(trains if i < len(cons) * 0.9 else tests).append(f'0\t{text}')

write_lines('data/pc_train.txt', trains)
write_lines('data/pc_test.txt',  tests)
