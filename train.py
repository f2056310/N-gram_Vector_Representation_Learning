from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--ndl',   required=True)
parser.add_argument('--input', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--aug',   required=True, type=int)
args   = parser.parse_args()

lat     = utils.Lattice(args.ndl)
maxsize = args.aug if args.aug > 0 else None
ckpt    = ModelCheckpoint(args.model, save_best_only=True)
early   = EarlyStopping(monitor='val_loss', patience=3)

x_train, y_train = utils.create_data(lat, maxsize, args.input)
model = utils.create_model(y_train.shape[1])
model.fit(
	x_train, y_train, validation_split=0.1,
	batch_size=128, epochs=10000,
	callbacks=[ckpt, early]
)
