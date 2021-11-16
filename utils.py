from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dropout, Dense

import numpy as np
import re
import struct

def read_lines(filename, encoding='utf-8'):
	lines = []
	with open(filename, 'r', encoding=encoding) as f:
		for line in f:
			line = line.rstrip()
			if line == '':
				continue
			lines.append(line)
	return lines

def split(text):
	ss = []
	for s in text.split(' '):
		if s == '-' or '-' not in s:
			ss.append(s)
		else:
			ss.extend(s.split('-'))
	return [s for s in ss if s != '']

def tidy(text):
	text = text.lower()
	text = re.sub(' *\u2026 *', ' ', text)
	text = re.sub("\\/", '/', text)
	text = re.sub("``", '"', text)
	text = re.sub("''", '"', text)
	text = re.sub(" '(d|ll|m|re|s|t|ve) ", "'\\1 ", text)
	text = re.sub(" n't", "n't", text)
	text = re.sub('([a-z]),([a-z])', '\\1, \\2', text)
	text = re.sub('([a-z])/([a-z])', '\\1 / \\2', text)
	text = re.sub('([a-z])\(([a-z])', '\\1 (\\2', text)
	text = re.sub('([a-z])\)([a-z])', '\\1) \\2', text)
	text = re.sub('([a-z])\\.\\.+([a-z])', '\\1. \\2', text)
	return text

def split_symbols(text):
	xs = []
	i  = 0
	while i < len(text):
		j = i
		while j < len(text):
			cat = unicodedata.category(text[j])
			if cat in ('Zs', 'Zl', 'Zp', 'Cc', 'Cf'):
				break
			j += 1
		if i < j:
			x = text[i : j]
			while len(x) > 0:
				cat = unicodedata.category(x[0])
				if cat[0] == 'P':
					xs.append(x[0])
					x = x[1 :]
			ys = []
			while len(x) > 0:
				cat = unicodedata.category(x[-1])
				if cat[0] == 'P':
					ts.append(x[-1])
					x = x[: -1]
			if x.endswith('\'s'):
				xs.append(x[: -2])
				xs.append(x[-2 :])
			elif len(x) > 0:
				xs.append(x)
			if len(ys) > 0:
				ys.reverse()
				xs.extend(ys)
		i = j + 1
	return ' '.join(xs)

def write_lines(filename, lines):
	with open(filename, 'w') as f:
		for line in lines:
			print(line, file=f)

def load_vecs(path):
	with open(path, 'rb') as f:
		content = f.read()

	max_v, max_d = 0, 0
	i = 0
	while content[i] != 0x20:
		max_v = max_v * 10 + content[i] - ord('0')
		i += 1
	i += 1
	while content[i] != 0x0a:
		max_d = max_d * 10 + content[i] - ord('0')
		i += 1
	i += 1
	
	f = 'f' * max_d
	surfs, vecs = [], []
	for v in range(max_v):
		surf = []
		while content[i] != 0x20:
			surf.append(content[i])
			i += 1
		i += 1
		vec  = struct.unpack_from(f, content, i)
		i += 4 * max_d
		if content[i] != 0x0a:
			raise
		i += 1
		surf = struct.pack('B' * len(surf), *surf).decode('utf-8')
		vec  = np.array(vec, dtype='float32')
		len2 = np.dot(vec, vec.T)
		if len2 == 0:
			continue
		surfs.append(surf)
		vecs .append(vec / np.sqrt(len2))
	
	return (surfs, np.array(vecs))

class Lattice:
	def __init__(self, path):
		surfs, self.vecs = load_vecs(path)
		self.idxs = {surf: idx for idx, surf in enumerate(surfs)}
	
	def backward(self, x, z, seqs, seq):
		if x == z:
			seqs.append(tuple(seq))
		elif x < z:
			for (idx, y) in self.paths[x]:
				seq.append(idx)
				self.backward(y, z, seqs, seq)
				seq.pop() 
	
	def analyze0(self, z):
		seqss = [set() for _ in range(z + 1)]
		seqss[0].add(())
		for y in range(1, z + 1):
			for n in range(1, 6):
				if y - n < 0:
					break
				seqs = []
				self.backward(y - n, y, seqs, [])
				for seq1 in seqss[y - n]:
					for seq2 in seqs:
						seqss[y].add(seq1 + seq2)
		return list(seqss[z])
	
	def analyze1(self, z, ns):
		seq = []
		x   = 0
		while x < z:
			ts, ps = [], []
			for (idx, y) in self.paths[x]:
				ts.append((idx, y))
				ps.append(ns[y] / ns[x])
			(idx, y) = ts[np.random.choice(len(ts), p=ps)]
			seq.append(idx)
			x = y
		return tuple(seq)
	
	def analyze(self, ss, maxsize):
		z = len(ss)
		
		self.paths = [[] for _ in range(z)]
		for i in range(z):
			for n in range(1, 6):
				if i + n > z or maxsize is None and n > 1:
					break
				surf = '\t'.join(ss[i : i + n])
				if surf in self.idxs:
					self.paths[i].append((self.idxs[surf], i + n))
				elif n == 1:
					self.paths[i].append((-1,              i + n))
		
		ns = [0] * (z + 1)
		ns[z] = 1
		for x in reversed(range(z)):
			ns[x] = sum(ns[y] for (idx, y) in self.paths[x])
		
		if maxsize is None or ns[0] < maxsize * 3:
			seqs = self.analyze0(z) # enumerate all
			np.random.shuffle(seqs)
			return seqs[: maxsize]
		else:
			seqs = set()
			while len(seqs) < maxsize:
				seqs.add(self.analyze1(z, ns)) # sample at random
			return list(seqs)

class Converter:
	def __init__(self, lat, maxsize, n_classes, n_maxlen):
		self.lat       = lat
		self.maxsize   = maxsize
		self.n_classes = n_classes
		self.n_maxlen  = n_maxlen
	
	def convert(self, label, ss):
		xys = []
		ys  = np.zeros((self.n_classes,), dtype='float32')
		ys[label] = 1
		for idxs in self.lat.analyze(ss, maxsize=self.maxsize):
			xs = np.zeros((self.n_maxlen, 300), dtype='float32')
			for i, idx in enumerate(idxs):
				if idx >= 0:
					xs[i] = self.lat.vecs[idx]
			xys.append((xs, ys))
		return xys

def prepare(filename):
	tups = []
	with open(filename, 'r') as f:
		for line in f:
			line = line.rstrip()
			if line == '':
				continue
			label, text = line.split('\t')
			tups.append((int(label), split(text)))
	
	n_classes = len(set([label for (label, ss) in tups]))
	n_maxlen  = max(len(ss) for (label, ss) in tups)
	
	return tups, n_classes, n_maxlen

def create_data(lat, maxsize, filename):
	tups, n_classes, n_maxlen = prepare(filename)
	
	conv = Converter(lat, maxsize, n_classes, n_maxlen)
	xys  = []
	for (label, ss) in tups:
		xys.extend(conv.convert(label, ss))
	
	np.random.shuffle(xys)
	return np.array([xs for (xs, ys) in xys]), np.array([ys for (xs, ys) in xys])

def create_model(n_classes):
	drop = Dropout(0.5)
	
	x = Input(shape=(None, 300), dtype='float32')
	
	h = x
	h = Bidirectional(LSTM(64, return_sequences=True))(h)
	h = drop(h)
	h = Bidirectional(LSTM(32))(h)
	h = drop(h)
	h = Dense(20, activation='relu')(h)
	h = Dense(n_classes, activation='softmax', kernel_initializer='normal')(h)
	y = h
	
	model = Model(x, y)
	model.compile('adam', 'categorical_crossentropy', metrics=['acc'])
	return model
