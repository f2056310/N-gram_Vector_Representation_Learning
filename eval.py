from tensorflow.keras.models import load_model

import argparse
import numpy as np
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--ndl',   required=True)
parser.add_argument('--input', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--aug',   required=True, type=int)
args   = parser.parse_args()

lat       = utils.Lattice(args.ndl)
model     = load_model(args.model)
maxsize   = args.aug if args.aug > 0 else None
n_classes = model.output.shape[-1]
tups, _, n_maxlen = utils.prepare(args.input)
conv      = utils.Converter(lat, maxsize, n_classes, n_maxlen)

cs  = np.zeros((n_classes,), dtype='int32')
xss = []
ts  = []
for (label, ss) in tups:
	xys = conv.convert(label, ss)
	for (xs, ys) in xys:
		xss.append(xs)
	ts.append((len(xys), label))

zss = model.predict(np.array(xss))
zs  = np.argmax(zss, axis=-1)

o, x = 0, 0
start = 0
for (n, label) in ts:
	cs.fill(0)
	for z in zs[start : start + n]:
		cs[z] += 1
	start += n
	if np.argmax(cs) == label:
		o += 1
	else:
		x += 1

print(f'acc={o / (o+x)}')
