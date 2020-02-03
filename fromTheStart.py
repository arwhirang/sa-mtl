import time, argparse, gc
import numpy as np
from sklearn import metrics
from rdkit import Chem
from feature import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import os
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("current pid:", os.getpid())

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("should be ok...right?")
    except RuntimeError as e:
        print(e)
else:
    print("gpu unlimited?")
START = time.time()

# hyperparameters ===================================
atomInfo = 21
structInfo = 21
lensize = atomInfo + structInfo
parser = argparse.ArgumentParser(description='CNN fingerprint')
parser.add_argument('--batchsize', '-b', type=int, default=32, help='Number of moleculars in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=51, help='Number of sweeps over the dataset to train')
parser.add_argument('--input', '-i', default='./TOX21', help='Input SDFs Dataset')
parser.add_argument('--atomsize', '-a', type=int, default=400, help='max length of smiles')
parser.add_argument('--protein', '-p', default="NR-AR", help='Name of protein (subdataset)')

parser.add_argument('--n_hid', type=int, default=264, help='No. of hidden perceptron')
parser.add_argument('--n_out', type=int, default=1, help='No. of output perceptron (class)')
args = parser.parse_args()


# -------------------------------------------------------------
# detaset function definition
def random_list(x, seed=0):
    np.random.seed(seed)
    np.random.shuffle(x)


def posNegNums(ydata):
    cntP = 0
    cntN = 0
    for ele in ydata:
        if ele == 1:
            cntP += 1
        else:
            cntN += 1
    return cntP, cntN


def makeData(proteinName):
    # load data =========================================
    print('start loading train data')
    afile = args.input + '/' + proteinName + '_wholetraining.smiles'
    smi = Chem.SmilesMolSupplier(afile, delimiter=' ', titleLine=False)  # smi var will not be used afterwards
    mols = [mol for mol in smi if mol is not None]

    # Make Feature Matrix ===============================
    F_list, T_list = [], []
    for mol in mols:
        if len(Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True)) > args.atomsize:
            print("too long mol was ignored")
        else:
            F_list.append(mol_to_feature(mol, -1, args.atomsize))
            T_list.append(mol.GetProp('_Name'))

    # Setting Dataset to model ==========================
    scaler = StandardScaler()
    F_list_scaled = scaler.fit_transform(F_list)
    F_list_scaled = np.clip(F_list_scaled, -5, 5)

    pos_num, neg_num = posNegNums(T_list)

    random_list(F_list_scaled)
    random_list(T_list)

    train_x, test_x, train_y, test_y = train_test_split(F_list_scaled, T_list, test_size=0.1)
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1111)

    train_y = np.asarray(train_y, dtype=np.int32).reshape(-1)
    train_x = np.asarray(train_x, dtype=np.float32).reshape(-1, args.atomsize * lensize)
    train_tf = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(args.batchsize)
    valid_y = np.asarray(valid_y, dtype=np.int32).reshape(-1)
    valid_x = np.asarray(valid_x, dtype=np.float32).reshape(-1, args.atomsize * lensize)
    valid_tf = tf.data.Dataset.from_tensors((valid_x, valid_y))  # no batch for validation sets
    return train_tf, valid_tf, pos_num, neg_num


train_tf, valid_tf, pos_num, neg_num = makeData("NR-AR")


def make_model(output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = keras.Sequential([
        keras.layers.Dense(400, activation='relu', input_shape=(args.atomsize * lensize,)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),
    ])

    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3), loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.AUC(name='auc'), keras.metrics.BinaryAccuracy(), keras.metrics.Mean(),
                           keras.metrics.Precision(name='precFunc'), keras.metrics.Recall(name='recallFunc')])

    return model


EPOCHS = 100
BATCH_SIZE = 64

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc',
                                                  verbose=1,
                                                  patience=10,
                                                  mode='max',
                                                  restore_best_weights=True)
initial_bias = np.log([pos_num / neg_num])
model = make_model(output_bias=initial_bias)
for X, Y in train_tf:
    results = model.evaluate(X, Y, batch_size=BATCH_SIZE, verbose=0)

initP = pos_num / (pos_num + neg_num)
expectedInitLoss = -np.log(initP) * initP - (1 - initP) * np.log(1 - initP)
print("Loss: {:0.4f}".format(results[0]), "expected Loss: ", expectedInitLoss)

initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
model.save_weights(initial_weights)

model.load_weights(initial_weights)
baseline_history = model.fit(
    train_tf,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    validation_data=valid_tf)


