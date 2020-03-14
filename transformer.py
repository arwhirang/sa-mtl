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
parser.add_argument('--num_layers', type=int, default=5, help='No. of hidden perceptron')
parser.add_argument('--d_model', type=int, default=100, help='No. of hidden perceptron')#default 512
parser.add_argument('--dff', type=int, default=1024, help='No. of hidden perceptron')
parser.add_argument('--num_heads', type=int, default=5, help='No. of hidden perceptron')
parser.add_argument('--dropout_rate', '-d', type=float, default=0.5, help='No. of hidden perceptron')
parser.add_argument('--lr', '-l', type=float, default=0.00005, help='No. of hidden perceptron')
parser.add_argument('--max_vocab_size', type=int, default=1025, help='No. of output perceptron (class)')
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
train_tf = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(args.batchsize)#batchsize, 200
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
    #train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1111)

    train_y = np.asarray(train_y, dtype=np.int32).reshape(-1)
    train_x = np.asarray(train_x, dtype=np.float32).reshape(-1, args.atomsize, lensize)
    pos_num, neg_num = posNegNums(train_y)
    train_tf = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(args.batchsize)
    #valid_y = np.asarray(valid_y, dtype=np.int32).reshape(-1)
    #valid_x = np.asarray(valid_x, dtype=np.float32).reshape(-1, args.atomsize, lensize)
    #valid_tf = tf.data.Dataset.from_tensor_slices((valid_x, valid_y)).batch(args.batchsize)
    test_y = np.asarray(test_y, dtype=np.int32).reshape(-1)
    test_x = np.asarray(test_x, dtype=np.float32).reshape(-1, args.atomsize, lensize)
    test_tf = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(args.batchsize)
    return train_tf, test_tf, pos_num, neg_num


#train_tf, test_tf, pos_num, neg_num = makeData_scfp("NR-AR")

def char2indices(listStr, dicC2I):
    listIndices = [0]*200
    charlist = listStr
    for i, c in enumerate(charlist):
        if c not in dicC2I:
            dicC2I[c] = len(dicC2I)
            listIndices[i] = dicC2I[c]
        else:
            listIndices[i] = dicC2I[c]
    return listIndices

def makeDataForSmilesOnly(proteinName, dicC2I_):
    listX, listY = [], []
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
        listX.append(char2indices(splitted[0], dicC2I_))#length can vary
        listY.append(float(splitted[1]))
    f.close()
    print("how many weird cases exist?", cntTooLong, weirdButUseful)
    random_list(listX)
    random_list(listY)
    train_x, test_x, train_y, test_y = train_test_split(listX, listY, test_size=0.1)
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1111)
    pos_num, neg_num = posNegNums(train_y)    
    print("pos_num, neg_num", pos_num, neg_num)
    train_tf = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(args.batchsize)
    valid_tf = tf.data.Dataset.from_tensor_slices((valid_x, valid_y)).batch(args.batchsize)
    test_tf = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(args.batchsize)
    return train_tf, valid_tf, test_tf, pos_num, neg_num

dicC2I = {}
train_tf, valid_tf, test_tf, pos_num, neg_num = makeDataForSmilesOnly(args.proteinTarget, dicC2I)


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
        tf.keras.layers.Dense(dff_, activation="relu"),#dff_, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model_)#d_model_)  # (batch_size, seq_len, d_model)
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

class Encoder(tf.keras.Model):
    def __init__(self, num_layers_, d_model_, num_heads_, dff_, maximum_position_encoding, output_bias, seq_len, rate=0.1): #input_vocab and max_vocab are the same
        super(Encoder, self).__init__()
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        self.seq_size = seq_len
        self.d_model = d_model_
        self.num_layers = num_layers_
        #self.embedding = tf.keras.layers.Embedding(maximum_position_encoding, d_model_, mask_zero=True)
        #self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        self.enc_layers = [EncoderLayer(d_model_, num_heads_, dff_, rate) for _ in range(num_layers_)]
        #self.enc_layers = EncoderLayer(d_model_, num_heads_, dff_, rate)
        self.dropout = tf.keras.layers.Dropout(rate)
        #self.semi_final = tf.keras.layers.RNN(tf.keras.layers.GRUCell(args.n_hid, recurrent_initializer='glorot_uniform'))
        self.semi_final = tf.keras.layers.Dense(1)
        
        self.gru = tf.keras.layers.GRU(self.d_model, return_sequences=True)
        self.bidrect = tf.keras.layers.Bidirectional(self.gru, merge_mode='sum')
        #self.pads1 = tf.constant([[0, 0], [0, 5-1], [0, 0]])
        #self.conv1 = tf.keras.layers.Conv2D(self.d_model, [5, self.d_model], strides=1)
        #self.maxp = tf.keras.layers.MaxPool1D([196])
        self.final_layer = tf.keras.layers.Dense(args.n_out, bias_initializer=output_bias)#, activation='sigmoid'

    def call(self, x_, training, mask_att, justmask):
        #seq_len = tf.shape(x_)[1]
        #x_.set_shape([None, self.seq_size])
        # adding embedding and position encoding.
        #x_ = self.embedding(inputs)  # (batch_size, input_seq_len, d_model)
        #mask_ = self.embedding.compute_mask(inputs)
        #x_ *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        #x_ += self.pos_encoding[:, :seq_len, :]
        #x_ = self.dropout(x_, training=training)## <= reached 85

        #x_ = tf.keras.layers.Reshape([seq_len, self.d_model, 1])(x_)
        #x_ = self.conv1(x_)
        #x_ = tf.keras.layers.Reshape([seq_len-5+1, self.d_model])(x_)
        #x_ = tf.pad(x_, self.pads1)#shape (batch, 200, d_model)
        x_ = self.gru(x_, mask=justmask)
        #x_ = self.bidrect(x_)
        for i in range(self.num_layers):
            x_ = self.enc_layers[i](x_, training, mask_att)

        #x_ = self.enc_layers(x_, training, mask_att)
        # x shape (batch_size, input_seq_len, d_model)
        ##x_ = self.conv1(x_) # (batch_size, seq_len-5+1, 1024)
        x_ = self.dropout(x_, training=training) ##<= reached 89
        out = self.semi_final(x_)
        out = tf.keras.layers.Reshape([self.seq_size])(out)#since final layer has dimension size of 1
        #out = self.dropout(out, training=training) ##<= reached 88
        out = self.final_layer(out) # (batch_size, 1)
        out = tf.squeeze(out, axis=[1])
        return out, tf.math.sigmoid(out)


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


learning_rate = args.lr#0.0001#CustomSchedule(args.d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
#loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM)

train_loss = tf.keras.metrics.Mean(name='train_loss')
AUCFunc = tf.keras.metrics.AUC()
accFunc = tf.keras.metrics.BinaryAccuracy()
precFunc = tf.keras.metrics.Precision(name='precFunc')
recallFunc = tf.keras.metrics.Recall(name='recallFunc')
initial_bias = np.log([pos_num / neg_num])
#weight_for_0 = tf.convert_to_tensor((1 / neg_num)*(pos_num + neg_num)/2.0, dtype=tf.float32)#not used 
weight_for_1 = tf.convert_to_tensor((1 / pos_num)*(pos_num + neg_num)/2.0, dtype=tf.float32)

encoder = Encoder(args.num_layers, args.d_model, args.num_heads, args.dff, len(dicC2I), output_bias=initial_bias, seq_len=200, rate=args.dropout_rate)


def loss_function(real, pred_logit, sampleW=None):
    #loss_ = loss_object(real, pred)
    cross_ent = tf.nn.weighted_cross_entropy_with_logits(logits=pred_logit, labels=real, pos_weight=sampleW)
    return tf.reduce_mean(cross_ent)


def create_padding_mask_fp2vec(seq):
    baseseq = tf.math.equal(seq, 0)
    seq = tf.cast(baseseq, tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :], tf.math.logical_not(baseseq)# (batch_size, 1, 1, seq_len)


def create_padding_mask_scfp(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    seq = tf.cast(tf.math.argmin(seq, axis=-1), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :], seq# (batch_size, 1, 1, atom_size)


# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]

bit_size = 1024 # circular fingerprint
emb = tf.Variable(tf.random.uniform([bit_size, args.d_model], -1, 1), dtype=tf.float32)
pads = tf.constant([[1,0], [0,0]])
embeddings_ = tf.pad(emb, pads)#1025, 200

#tf.function(input_signature=train_step_signature)
def train_step(inp_, real):  # shape is [batch, seq_len]
    inp_padding_mask, justmask = create_padding_mask_fp2vec(inp_)
    with tf.GradientTape() as tape:
        #predictions, _ = transformer(inp_, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
        pred_logit_, pred = encoder(tf.nn.embedding_lookup(embeddings_, inp_), True, inp_padding_mask, justmask)
        #pred_logit_, pred = encoder(inp_, True, inp_padding_mask, justmask)
        loss = loss_function(real, pred_logit_, sampleW=weight_for_1)
    gradients = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))
    train_loss(loss)


def eval_step(inp_, real):
    inp_padding_mask, justmask = create_padding_mask_fp2vec(inp_)
    _, pred = encoder(tf.nn.embedding_lookup(embeddings_, inp_), False, inp_padding_mask, justmask)
    #_, pred = encoder(inp_, False, inp_padding_mask, justmask)
    precFunc.update_state(y_true=real, y_pred=pred)
    recallFunc.update_state(y_true=real, y_pred=pred)
    AUCFunc.update_state(y_true=real, y_pred=pred)
    accFunc.update_state(y_true=real, y_pred=pred)


checkpoint_dir = "tr0/cp.ckpt"
#checkpoint_dir = os.path.dirname(checkpoint_path)
f2w = open("performance", "w")

bestAUC = 0
bestEpoch = 0
encoder.save_weights(checkpoint_dir)
for epoch in range(args.epochs):
    start = time.time()
    train_loss.reset_states()
    precFunc.reset_states()
    recallFunc.reset_states()
    AUCFunc.reset_states()
    accFunc.reset_states()
    
    for (batch, (X, Y)) in enumerate(train_tf):
        train_step(X, Y)

    for (batch, (X, Y)) in enumerate(valid_tf):
        eval_step(X, Y)

    if bestAUC < AUCFunc.result():
        bestAUC = AUCFunc.result()
        bestEpoch = epoch + 1
        encoder.save_weights(checkpoint_dir)
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, checkpoint_dir))
    print('Train Epoch {} Loss {:.4f}'.format(epoch + 1, train_loss.result()))
    print('Valid prec {:.4f} recall {:.4f} AUC {:.4f}, acc {:.4f}'.format(precFunc.result(), recallFunc.result(),
                                                                          AUCFunc.result(), accFunc.result()))
    f2w.write('Train Epoch {} Loss {:.4f}'.format(epoch + 1, train_loss.result()))
    f2w.write("\n")
    f2w.write('Valid prec {:.4f} recall {:.4f} AUC {:.4f}, acc {:.4f}'.format(precFunc.result(), recallFunc.result(),
                                                                          AUCFunc.result(), accFunc.result()))
    f2w.write("\n")
    f2w.flush()
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))



######testing phase
precFunc.reset_states()
recallFunc.reset_states()
AUCFunc.reset_states()
accFunc.reset_states()

encoder.load_weights(checkpoint_dir)
print("loaded the weights from the best epoch:", bestEpoch)

for (batch, (X, Y)) in enumerate(test_tf):
    eval_step(X, Y)

print('Test prec {:.4f} recall {:.4f} AUC {:.4f}, acc {:.4f}'.format(precFunc.result(), recallFunc.result(), AUCFunc.result(), accFunc.result()))

f2w.close()
