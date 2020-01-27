import tensorflow as tf

from trainer import Trainer#not called while making training works
from models.GraphGANModel import GraphGANModel
from models.layers import encoder_rgcn, decoder_adj
from optimizer import GraphGANOptimizer
import os
import numpy as np
from sklearn.model_selection import train_test_split
from rdkit import Chem
from feature import *
import time

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

tf.compat.v1.disable_v2_behavior()


def random_list(x, seed=0):
    np.random.seed(seed)
    np.random.shuffle(x)


# load data =========================================
print('start loading train data')
afile = 'TOX21/NR-AR_wholetraining.smiles'
smi = Chem.SmilesMolSupplier(afile, delimiter=' ', titleLine=False)
mols_ = [mol for mol in smi if mol is not None]
atomsize = 400
batch_dim = 32
la = 1
dropout = 0
n_critic = 5
metric = 'validity, sas'
n_samples = 5000
z_dim = 8
epochs = 10
save_every = None

# Make Feature Matrix ===============================
F_list, T_list = [], []
for mol in mols_:
    if len(Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True)) > atomsize:
        print("too long mol was ignored")
    else:
        F_list.append(mol_to_feature(mol, -1, atomsize))
        T_list.append(mol.GetProp('_Name'))

# Setting Dataset to model ==========================
random_list(F_list)
random_list(T_list)

train_x, test_x, train_y, test_y = train_test_split(F_list, T_list, test_size=0.1)
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1111)

train_y = np.asarray(train_y, dtype=np.int32).reshape(-1)
train_x = np.asarray(train_x, dtype=np.float32).reshape(-1, atomsize, lensize)
train_tf = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(batch_dim)

valid_x = np.asarray(valid_x, dtype=np.int32).reshape(-1)
valid_y = np.asarray(valid_y, dtype=np.float32).reshape(-1, atomsize, lensize)
valid_tf = tf.data.Dataset.from_tensor_slices((valid_x, valid_y)).batch(batch_dim)
steps = (len(train_y) // batch_dim)


def train_fetch_dict(i, _optimizer):
    a = [_optimizer.train_step_G] if i % n_critic == 0 else [_optimizer.train_step_D]
    b = [_optimizer.train_step_V] if i % n_critic == 0 and la < 1 else []
    return a + b


def train_feed_dict(i, train_x, train_y, _epoch, _model, _optimizer, _batch_dim, _dropout):
    embeddings = _model.sample_z(_batch_dim)
    if la < 1:
        if i % n_critic == 0:
            # rewardR = reward(train_x, train_y)
            # lga = session.run([_model.logit_gumbel_argmax],
            #                    feed_dict={_model.training: False, _model.embeddings: embeddings})
            # lga = np.argmax(lga, axis=-1)
            # newmols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]
            # rewardF = reward(newmols, train_y)
            feed_dict = {_model.input2gen: train_x,
                         _model.embeddings: embeddings,
                         # _model.rewardR: rewardR,
                         # _model.rewardF: rewardF,
                         _model.training: True,
                         _model.dropout_rate: _dropout,
                         _optimizer.la: la if _epoch > 0 else 1.0}
        else:
            feed_dict = {_model.input2gen: train_x,
                         _model.embeddings: embeddings,
                         _model.training: True,
                         _model.dropout_rate: _dropout,
                         _optimizer.la: la if _epoch > 0 else 1.0}
    else:
        feed_dict = {_model.input2gen: train_x,
                     _model.embeddings: embeddings,
                     _model.training: True,
                     _model.dropout_rate: _dropout,
                     _optimizer.la: 1.0}
    return feed_dict


# def eval_fetch_dict(i, epochs, min_epochs, model, optimizer):
#     return {'loss D': optimizer.loss_D, 'loss G': optimizer.loss_G, 'loss RL': optimizer.loss_RL,
#             'loss V': optimizer.loss_V, 'la': optimizer.la}
#
#
# def eval_feed_dict(i, epochs, min_epochs, model, optimizer, batch_dim):
#     # mols, _, _, a, x, _, _, _, _ = data.next_validation_batch()
#     embeddings = model.sample_z(a.shape[0])
#     rewardR = reward(mols)
#     n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
#                        feed_dict={model.training: False, model.embeddings: embeddings})
#     n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
#     # mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]
#     rewardF = reward(mols)
#     feed_dict = {model.edges_labels: a,
#                  model.nodes_labels: x,
#                  model.embeddings: embeddings,
#                  model.rewardR: rewardR,
#                  model.rewardF: rewardF,
#                  model.training: False}
#     return feed_dict
#
#
# def test_fetch_dict(model, optimizer):
#     return {'loss D': optimizer.loss_D, 'loss G': optimizer.loss_G, 'loss RL': optimizer.loss_RL,
#             'loss V': optimizer.loss_V, 'la': optimizer.la}
#
#
# def test_feed_dict(model, optimizer, batch_dim):
#     # mols, _, _, a, x, _, _, _, _ = data.next_test_batch()
#     embeddings = model.sample_z(a.shape[0])
#     rewardR = reward(mols)
#     n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
#                        feed_dict={model.training: False, model.embeddings: embeddings})
#     n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
#     # mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]
#     rewardF = reward(mols)
#     feed_dict = {model.edges_labels: a,
#                  model.nodes_labels: x,
#                  model.embeddings: embeddings,
#                  model.rewardR: rewardR,
#                  model.rewardF: rewardF,
#                  model.training: False}
#     return feed_dict


# def reward(mols):
#     rr = 1.
    # rr *= MolecularMetrics.natural_product_scores(mols, norm=True)
    # return rr.reshape(-1, 1)


# def _eval_update(i, epochs, min_epochs, model, optimizer, batch_dim, eval_batch):
#     # mols = samples(data, model, session, model.sample_z(n_samples), sample=True)
#     # m0, m1 = all_scores(mols, data, norm=True)
#     m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
#     m0.update(m1)
#     return m0
#
#
# def _test_update(model, optimizer, batch_dim, test_batch):
#     # mols = samples(data, model, session, model.sample_z(n_samples), sample=True)
#     # m0, m1 = all_scores(mols, data, norm=True)
#     m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
#     m0.update(m1)
#     return m0


# model
model = GraphGANModel(atomsize, lensize,
                      z_dim,
                      decoder_units=(128, 256, 256),
                      discriminator_units=((128, 64), 128, (128, 64)),
                      decoder=decoder_adj,
                      discriminator=encoder_rgcn,
                      batch_discriminator=False)

# optimizer
optimizer = GraphGANOptimizer(model, learning_rate=1e-3, feature_matching=False)
# session
session = tf.Session()
session.run(tf.global_variables_initializer())


def train_step(_step, _epoch, _model, _optimizer, _batch_dim, _dropout):
    return session.run(train_fetch_dict(_step, _optimizer),
                       feed_dict=train_feed_dict(_step, _epoch, _model, _optimizer, _batch_dim, _dropout))


start_time = time.time()

for epoch in range(epochs + 1):
    if epoch < epochs:
        for step in range(steps):
            train_step(steps * epoch + step, epoch, model, optimizer, batch_dim, dropout)

