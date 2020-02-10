from __future__ import absolute_import, division, print_function, unicode_literals

# try:
#     !pip install tf-nightly
# except Exception:
#     pass
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from rdkit import Chem
from feature import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

parser = argparse.ArgumentParser(description='argparse')
parser.add_argument('--batchsize', '-b', type=int, default=51, help='Number of moleculars in each mini-batch')
parser.add_argument('--epochs', '-e', type=int, default=50, help='Number of sweeps over the dataset to train')
parser.add_argument('--input', '-i', default='./TOX21', help='Input SDFs Dataset')
parser.add_argument('--proteinTarget', '-p', required=True, help='target data')
parser.add_argument('--lastDim', '-a', type=int, default=42, help='max length of smiles')
parser.add_argument('--num_layers', type=int, default=6, help='No. of hidden perceptron')
parser.add_argument('--d_model', type=int, default=42, help='No. of hidden perceptron')  # default 512
parser.add_argument('--dff', type=int, default=2048, help='No. of hidden perceptron')
parser.add_argument('--num_heads', type=int, default=7, help='No. of hidden perceptron')
parser.add_argument('--dropout_rate', '-d', type=float, default=0.3, help='No. of hidden perceptron')
parser.add_argument('--lr', '-l', type=float, default=0.00005, help='No. of hidden perceptron')
parser.add_argument('--max_vocab_size', type=int, default=1026, help='No. of output perceptron (class)')
parser.add_argument('--atomsize', '-c', type=int, default=400, help='max length of smiles')
parser.add_argument('--seq_size', '-s', type=int, default=200, help='seq length of smiles fp2vec')
parser.add_argument('--n_hid', type=int, default=256, help='No. of hidden perceptron')
parser.add_argument('--n_out', type=int, default=1, help='No. of output perceptron (class)')
args = parser.parse_args()
"""
print('start loading data for fp2vec')
dataX = np.load('tox21_fp.npy')#7439, 1024
dataY_concat  = np.load('tox21_Y.npy', allow_pickle=True)#12, 7439
index = np.load('tox21_index.npy', allow_pickle=True)#12, 7439 <= 7439 is just for the first dataset - every dataset has different size.
print("dataX.shape:", dataX.shape)#8014, 1024
dataX3 = [dataX[i] for i in index[2]]#NR-AR


def prepro_X(inpX):
    data_x = []
    for i in range(len(inpX)):
        fp = [0] * args.seq_size
        n_ones = 0
        for j in range(1024):#inpX shape = variable, 1024
            if inpX[i][j] == 1:
                fp[n_ones] = j + 1
                n_ones += 1
        data_x.append(fp)
    return np.array(data_x, dtype=np.int32).reshape(-1, args.seq_size)#variable, 200


def prepro_Y(inpY):
    #data_y = [ele for ele in inpY]
    data_y = []
    for i in range(len(inpY)):
        data_y.append(inpY[i])
    return np.array(data_y, dtype=np.int32)#variable
"""


def posNegNums(ydata):
    cntP = 0
    cntN = 0
    for ele in ydata:
        if ele == 1:
            cntP += 1
        else:
            cntN += 1
    return cntP, cntN


# detaset function definition
def random_list(x, seed=0):
    np.random.seed(seed)
    np.random.shuffle(x)


"""
_dataX3, _dataY3 = prepro_X(dataX3), prepro_Y(dataY_concat[2])#NR-AR
random_list(_dataX3)
random_list(_dataY3)
pos_num, neg_num = posNegNums(_dataY3)
train_x, test_x, train_y, test_y = train_test_split(_dataX3, _dataY3, test_size=0.1)
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1111)
train_tf = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(args.batchsize)#batchsize, 200
valid_tf = tf.data.Dataset.from_tensor_slices((valid_x, valid_y)).batch(args.batchsize)
test_tf = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(args.batchsize)
"""


def makeData_scfp(proteinName):
    # load data =========================================
    print('start loading train data')
    afile = args.input + '/' + proteinName + '_wholetraining.smiles'

    #############
    #    afile = args.input + '/' + proteinName + '_fakelabels'
    #############

    smi = Chem.SmilesMolSupplier(afile, delimiter=' ', titleLine=False)  # smi var will not be used afterwards
    mols = [mol for mol in smi if mol is not None]

    ##############
    #    realY = []
    #    f = open('TOX21/NR-AR_wholetraining.smiles', 'r')
    #    lines = f.readlines()
    ##############

    # Make Feature Matrix ===============================
    F_list, T_list = [], []
    for i, mol in enumerate(mols):
        if len(Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True)) > args.atomsize:
            print("too long mol was ignored")
        else:
            F_list.append(mol_to_feature(mol, -1, args.atomsize))
            T_list.append(mol.GetProp('_Name'))

    ###############
    #            splitted = lines[i].split(" ")
    #            realY.append(int(splitted[1]))
    #    f.close()
    #    T_list = realY
    ###############

    # Setting Dataset to model ==========================
    scaler = StandardScaler()
    F_list_scaled = scaler.fit_transform(F_list)
    F_list_scaled = np.clip(F_list_scaled, -5, 5)
    random_list(F_list_scaled)
    random_list(T_list)
    train_x, test_x, train_y, test_y = train_test_split(F_list_scaled, T_list, test_size=0.1)
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1111)

    train_y = np.asarray(train_y, dtype=np.int32).reshape(-1)
    train_x = np.asarray(train_x, dtype=np.float32).reshape(-1, args.atomsize, lensize)
    pos_num, neg_num = posNegNums(train_y)
    train_tf = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(args.batchsize)
    valid_y = np.asarray(valid_y, dtype=np.int32).reshape(-1)
    valid_x = np.asarray(valid_x, dtype=np.float32).reshape(-1, args.atomsize, lensize)
    valid_tf = tf.data.Dataset.from_tensor_slices((valid_x, valid_y)).batch(args.batchsize)
    test_y = np.asarray(test_y, dtype=np.int32).reshape(-1)
    test_x = np.asarray(test_x, dtype=np.float32).reshape(-1, args.atomsize, lensize)
    test_tf = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(args.batchsize)
    return train_tf, valid_tf, test_tf, pos_num, neg_num


# train_tf, valid_tf, test_tf, pos_num, neg_num = makeData_scfp("NR-AR")

def char2indices(listStr, dicC2I):
    listIndices = [0] * 200
    charlist = listStr
    for i, c in enumerate(charlist):
        if c not in dicC2I:
            dicC2I[c] = len(dicC2I)
            listIndices[i] = dicC2I[c]
        else:
            listIndices[i] = dicC2I[c]
    return listIndices


def makeDataForSmilesOnly(proteinName):
    listX, listY = [], []
    dicC2I = {}
    afile = args.input + '/' + proteinName + '_wholetraining.smiles'
    f = open(afile, "r")
    lines = f.readlines()
    cntTooLong = 0
    weirdButUseful = 0
    for line in lines:
        splitted = line.split(" ")
        if len(splitted[0]) >= 200:
            cntTooLong += 1
            if splitted[1] == "1":
                weirdButUseful += 1
            continue
        listX.append(char2indices(splitted[0], dicC2I))  # length can vary
        listY.append(int(splitted[1]))
    f.close()
    # print("how many weird cases exist?", cntTooLong, weirdButUseful)
    random_list(listX)
    random_list(listY)
    train_x, test_x, train_y, test_y = train_test_split(listX, listY, test_size=0.1)
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1111)
    train_tf = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(args.batchsize)
    valid_tf = tf.data.Dataset.from_tensor_slices((valid_x, valid_y)).batch(args.batchsize)
    test_tf = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(args.batchsize)
    return train_tf, valid_tf, test_tf


train_tf1, valid_tf1, test_tf1 = makeDataForSmilesOnly("NR-AR-LBD")
train_tf2, valid_tf2, test_tf2 = makeDataForSmilesOnly("NR-AR")
train_tf3, valid_tf3, test_tf3 = makeDataForSmilesOnly("NR-AhR")
train_tf4, valid_tf4, test_tf4 = makeDataForSmilesOnly("NR-Aromatase")
train_tf5, valid_tf5, test_tf5 = makeDataForSmilesOnly("NR-ER-LBD")
train_tf6, valid_tf6, test_tf6 = makeDataForSmilesOnly("NR-ER")
train_tf7, valid_tf7, test_tf7 = makeDataForSmilesOnly("NR-PPAR-gamma")
train_tf8, valid_tf8, test_tf8 = makeDataForSmilesOnly("SR-ARE")
train_tf9, valid_tf9, test_tf9 = makeDataForSmilesOnly("SR-ATAD5")
train_tf10, valid_tf10, test_tf10 = makeDataForSmilesOnly("SR-HSE")
train_tf11, valid_tf11, test_tf11 = makeDataForSmilesOnly("SR-MMP")
train_tf12, valid_tf12, test_tf12 = makeDataForSmilesOnly("SR-p53")


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) but it must be broadcastable for addition.
    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k). Defaults to None.
    Returns:
      output, attention_weights
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model_, num_heads_):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads_
        self.d_model = d_model_
        assert d_model_ % self.num_heads == 0
        self.depth = d_model_ // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model_)
        self.wk = tf.keras.layers.Dense(d_model_)
        self.wv = tf.keras.layers.Dense(d_model_)
        self.dense = tf.keras.layers.Dense(d_model_)

    def split_heads(self, x_, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x_ = tf.reshape(x_, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x_, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights


def point_wise_feed_forward_network(d_model_, dff_):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff_, activation="relu"),  # dff_, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model_)  # d_model_)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model_, num_heads_, dff_, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model_, num_heads_)
        self.ffn = point_wise_feed_forward_network(d_model_, dff_)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2


"""
def get_angles(pos, i, d_model_):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model_))
    return pos * angle_rates


def positional_encoding(position, d_model_):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model_)[np.newaxis, :], d_model_)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)
"""


class CustomHot(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomHot, self).__init__()

    def call(self, inputs):
        return tf.one_hot(inputs, 12)


class CustomRSum(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomRSum, self).__init__()

    def call(self, inputs, dWhich):
        return tf.math.reduce_sum(inputs * dWhich, axis=1)  # only the 1 instnce survives


class Encoder(tf.keras.Model):
    def __init__(self, num_layers_, d_model_, num_heads_, dff_, maximum_position_encoding, seq_len,
                 rate=0.1):  # input_vocab and max_vocab are the same
        super(Encoder, self).__init__()
        self.seq_size = seq_len
        self.d_model = d_model_
        self.num_layers = num_layers_
        self.embedding = tf.keras.layers.Embedding(maximum_position_encoding, d_model_)
        # self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        self.enc_layers = [EncoderLayer(d_model_, num_heads_, dff_, rate) for _ in range(num_layers_)]
        # self.enc_layers = EncoderLayer(d_model_, num_heads_, dff_, rate)
        self.dropout = tf.keras.layers.Dropout(rate)
        # self.semi_final = tf.keras.layers.RNN(tf.keras.layers.GRUCell(args.n_hid, recurrent_initializer='glorot_uniform'))#bad
        self.semi_final = tf.keras.layers.Dense(1)
        self.finalFC1 = tf.keras.layers.Dense(self.n_out, activation='sigmoid')
        self.finalFC2 = tf.keras.layers.Dense(self.n_out, activation='sigmoid')
        self.finalFC3 = tf.keras.layers.Dense(self.n_out, activation='sigmoid')
        self.finalFC4 = tf.keras.layers.Dense(self.n_out, activation='sigmoid')
        self.finalFC5 = tf.keras.layers.Dense(self.n_out, activation='sigmoid')
        self.finalFC6 = tf.keras.layers.Dense(self.n_out, activation='sigmoid')
        self.finalFC7 = tf.keras.layers.Dense(self.n_out, activation='sigmoid')
        self.finalFC8 = tf.keras.layers.Dense(self.n_out, activation='sigmoid')
        self.finalFC9 = tf.keras.layers.Dense(self.n_out, activation='sigmoid')
        self.finalFC10 = tf.keras.layers.Dense(self.n_out, activation='sigmoid')
        self.finalFC11 = tf.keras.layers.Dense(self.n_out, activation='sigmoid')
        self.finalFC12 = tf.keras.layers.Dense(self.n_out, activation='sigmoid')

    def call(self, x_, whichClass, training, mask_att, justmask):
        seq_len = tf.shape(x_)[1]
        # x_.set_shape([None, self.seq_size])
        # adding embedding and position encoding.
        x_ = self.embedding(x_)  # (batch_size, input_seq_len, d_model)
        x_ *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # x_ += self.pos_encoding[:, :seq_len, :]
        x_ = self.dropout(x_, training=training)
        for i in range(self.num_layers):
            x_ = self.enc_layers[i](x_, training, mask_att)

        # x_ = self.enc_layers(x_, training, mask_att)
        # x shape (batch_size, input_seq_len, d_model)
        out = self.semi_final(x_)  # (batch_size, seq_len, 1)
        out = tf.keras.layers.Reshape([-1])(out)  # since final layer has dimension size of 1
        # out = self.dropout(x_, training=training)
        cl1 = self.finalFC1(out)
        cl2 = self.finalFC2(out)
        cl3 = self.finalFC3(out)
        cl4 = self.finalFC4(out)
        cl5 = self.finalFC5(out)
        cl6 = self.finalFC6(out)
        cl7 = self.finalFC7(out)
        cl8 = self.finalFC8(out)
        cl9 = self.finalFC9(out)
        cl10 = self.finalFC10(out)
        cl11 = self.finalFC11(out)
        cl12 = self.finalFC12(out)
        x_out = tf.keras.layers.concatenate([cl1, cl2, cl3, cl4, cl5, cl6, cl7, cl8, cl9, cl10, cl11, cl12])
        decideWhich = CustomHot()(whichClass)
        return CustomRSum()(x_out, decideWhich)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model_, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model_
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = args.lr  # 0.0001#CustomSchedule(args.d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate)  # , beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM)


def loss_function(real, pred, sampleW=None):
    # mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    # mask = tf.cast(mask, dtype=loss_.dtype)
    # loss_ *= mask
    return tf.reduce_mean(loss_)


train_loss = tf.keras.metrics.Mean(name='train_loss')
AUCFunc = tf.keras.metrics.AUC()
accFunc = tf.keras.metrics.BinaryAccuracy()
precFunc = tf.keras.metrics.Precision(name='precFunc')
recallFunc = tf.keras.metrics.Recall(name='recallFunc')
encoder = Encoder(args.num_layers, args.d_model, args.num_heads, args.dff, args.max_vocab_size,
                  seq_len=args.atomsize * 42, rate=args.dropout_rate)


def create_padding_mask_fp2vec(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :], seq  # (batch_size, 1, 1, seq_len)


def create_padding_mask_scfp(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    seq = tf.cast(tf.math.argmin(seq, axis=-1), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :], seq  # (batch_size, 1, 1, atom_size)


# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]


# @tf.function(input_signature=train_step_signature)
def train_step(inp_, real, whichClass):  # shape is [batch, seq_len]
    inp_padding_mask, justmask = create_padding_mask_fp2vec(inp_)
    with tf.GradientTape() as tape:
        # predictions, _ = transformer(inp_, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
        pred = encoder(inp_, whichClass, True, inp_padding_mask, justmask)
        loss = loss_function(real, pred)
    gradients = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))
    train_loss(loss)


def eval_step(inp_, real, whichClass):
    inp_padding_mask, justmask = create_padding_mask_fp2vec(inp_)
    pred = encoder(inp_, whichClass, False, inp_padding_mask, justmask)

    precFunc.update_state(y_true=real, y_pred=pred)
    recallFunc.update_state(y_true=real, y_pred=pred)
    AUCFunc.update_state(y_true=real, y_pred=pred)
    accFunc.update_state(y_true=real, y_pred=pred)


checkpoint_dir = "tr1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

bestAUC = 0
encoder.save_weights(checkpoint_dir)
for epoch in range(args.epochs):
    start = time.time()
    train_loss.reset_states()
    precFunc.reset_states()
    recallFunc.reset_states()
    AUCFunc.reset_states()
    accFunc.reset_states()

    for (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4), (X5, Y5), (X6, Y6), (X7, Y7), (X8, Y8), (X9, Y9), (X10, Y10), (
            X11, Y11), (X12, Y12) in zip(train_tf1, train_tf2, train_tf3, train_tf4, train_tf5, train_tf6, train_tf7,
                                         train_tf8, train_tf9, train_tf10, train_tf11, train_tf12):
        train_step(X1, Y1, 0)
        train_step(X2, Y2, 1)
        train_step(X3, Y3, 2)
        train_step(X4, Y4, 3)
        train_step(X5, Y5, 4)
        train_step(X6, Y6, 5)
        train_step(X7, Y7, 6)
        train_step(X8, Y8, 7)
        train_step(X9, Y9, 8)
        train_step(X10, Y10, 9)
        train_step(X11, Y11, 10)
        train_step(X12, Y12, 11)
    print('Train Epoch {} Loss {:.4f}'.format(epoch + 1, train_loss.result()))

    if epoch % 5 == 0:
        for (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4), (X5, Y5), (X6, Y6), (X7, Y7), (X8, Y8), (X9, Y9), (X10, Y10), (
                X11, Y11), (X12, Y12) in zip(valid_tf1, valid_tf2, valid_tf3, valid_tf4, valid_tf5, valid_tf6,
                                             valid_tf7, valid_tf8, valid_tf9, valid_tf10, valid_tf11, valid_tf12):
            eval_step(X1, Y1, 0)
            eval_step(X2, Y2, 1)
            eval_step(X3, Y3, 2)
            eval_step(X4, Y4, 3)
            eval_step(X5, Y5, 4)
            eval_step(X6, Y6, 5)
            eval_step(X7, Y7, 6)
            eval_step(X8, Y8, 7)
            eval_step(X9, Y9, 8)
            eval_step(X10, Y10, 9)
            eval_step(X11, Y11, 10)
            eval_step(X12, Y12, 11)

        if bestAUC < AUCFunc.result():
            bestAUC = AUCFunc.result()
            encoder.save_weights(checkpoint_dir)
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, checkpoint_dir))
        print('Valid prec {:.4f} recall {:.4f} AUC {:.4f}, acc {:.4f}'.format(precFunc.result(), recallFunc.result(),
                                                                              AUCFunc.result(), accFunc.result()))

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

######testing phase
precFunc.reset_states()
recallFunc.reset_states()
AUCFunc.reset_states()
accFunc.reset_states()

encoder.load_weights(checkpoint_dir)

for (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4), (X5, Y5), (X6, Y6), (X7, Y7), (X8, Y8), (X9, Y9), (X10, Y10), (
        X11, Y11), (X12, Y12) in zip(test_tf1, test_tf2, test_tf3, test_tf4, test_tf5, test_tf6,
                                     test_tf7, test_tf8, test_tf9, test_tf10, test_tf11, test_tf12):
    eval_step(X1, Y1, 0)
    eval_step(X2, Y2, 1)
    eval_step(X3, Y3, 2)
    eval_step(X4, Y4, 3)
    eval_step(X5, Y5, 4)
    eval_step(X6, Y6, 5)
    eval_step(X7, Y7, 6)
    eval_step(X8, Y8, 7)
    eval_step(X9, Y9, 8)
    eval_step(X10, Y10, 9)
    eval_step(X11, Y11, 10)
    eval_step(X12, Y12, 11)

print('Test prec {:.4f} recall {:.4f} AUC {:.4f}, acc {:.4f}'.format(precFunc.result(), recallFunc.result(),
                                                                     AUCFunc.result(), accFunc.result()))
