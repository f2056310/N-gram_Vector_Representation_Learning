from utils import *

import re
import xml.etree.ElementTree as ET

def repl(m):
	d = int(m[1])
	return chr(int(m[1])) + (' ' if d < 0x80 else '')

def clean(line):
	line = re.sub('<p class= .*?>', '', line)
	line = re.sub('<a href= .*?>',  '', line)
	line = re.sub('<i', '', line)
	line = re.sub('</a>', '', line)
	line = re.sub('</?em>', '', line)
	line = re.sub('&oacute ;', '\u00f3', line)
	line = re.sub('&#(\\d+) ; ', repl, line)
	line = re.sub('& ', '&amp; ', line)
	try:
		elem = ET.fromstring(f'<root>{line}</root>')
	except:
		line = re.sub('&', '&amp;', line)
		elem = ET.fromstring(f'<root>{line}</root>')
	text = elem.text
	if text is None:
		return ''
	text = tidy(text)
	text = split_symbols(text)
	return text

dirname = 'corpora/rotten_imdb'

subjs = read_lines(f'{dirname}/quote.tok.gt9.5000', 'Windows-1252')
objs  = read_lines(f'{dirname}/plot.tok.gt9.5000', 'Windows-1252')

trains, tests = [], []
for i, text in enumerate(subjs):
	text0 = text
	text = clean(text)
	if text == '':
		print(text0)
	(trains if i < len(subjs) * 0.9 else tests).append(f'1\t{text}')
for i, text in enumerate(objs):
	text = clean(text)
	(trains if i < len(objs)  * 0.9 else tests).append(f'0\t{text}')

write_lines('data/subj_train.txt', trains)
write_lines('data/subj_test.txt',  tests)
