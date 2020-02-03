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

parser = argparse.ArgumentParser(description='CNN fingerprint')
parser.add_argument('--batchsize', '-b', type=int, default=32, help='Number of moleculars in each mini-batch')
parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of sweeps over the dataset to train')
parser.add_argument('--input', '-i', default='./TOX21', help='Input SDFs Dataset')
parser.add_argument('--seq_size', '-a', type=int, default=200, help='max length of smiles')
parser.add_argument('--num_layers', type=int, default=6, help='No. of hidden perceptron')
parser.add_argument('--d_model', type=int, default=512, help='No. of hidden perceptron')
parser.add_argument('--dff', type=int, default=2048, help='No. of hidden perceptron')
parser.add_argument('--num_heads', type=int, default=8, help='No. of hidden perceptron')
parser.add_argument('--dropout_rate', type=int, default=0.1, help='No. of hidden perceptron')
parser.add_argument('--max_vocab_size', type=int, default=1026, help='No. of output perceptron (class)')

parser.add_argument('--n_hid', type=int, default=256, help='No. of hidden perceptron')
parser.add_argument('--n_out', type=int, default=1, help='No. of output perceptron (class)')
args = parser.parse_args()
"""
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']
tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)
tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)
sample_string = 'Transformer is awesome.'
tokenized_string = tokenizer_en.encode(sample_string)
print('Tokenized string is {}'.format(tokenized_string))
original_string = tokenizer_en.decode(tokenized_string)
print('The original string: {}'.format(original_string))
assert original_string == sample_string
for ts in tokenized_string:
    print('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))
BUFFER_SIZE = 20000
BATCH_SIZE = 64


def encode(lang1, lang2):
    lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(lang1.numpy()) + [tokenizer_pt.vocab_size + 1]
    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(lang2.numpy()) + [tokenizer_en.vocab_size + 1]
    return lang1, lang2


def tf_encode(pt, en):
    result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])
    return result_pt, result_en


train_dataset = train_examples.map(tf_encode)
# cache the dataset to memory to get a speedup while reading from it.
# noinspection SpellCheckingInspection
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.padded_batch(BATCH_SIZE)
pt_batch, en_batch = next(iter(val_dataset))
"""

print('start loading data')
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
    return np.array(data_x, dtype=np.int32)#variable, 200


def prepro_Y(inpY):
    #data_y = [ele for ele in inpY]
    data_y = []
    for i in range(len(inpY)):
        data_y.append([inpY[i]])
    return np.array(data_y, dtype=np.int32)#variable


_dataX3, _dataY3 = prepro_X(dataX3), prepro_Y(dataY_concat[2])#NR-AR
train_x3, test_x3, train_y3, test_y3 = train_test_split(_dataX3, _dataY3, test_size=0.1)
train_x3, valid_x3, train_y3, valid_y3 = train_test_split(train_x3, train_y3, test_size=0.1111)

#scaler = StandardScaler()
#train_features = scaler.fit_transform(train_x3)
#val_features = scaler.transform(valid_x3)
#test_features = scaler.transform(test_x3)


def random_list(x, y, seed=0):
    np.random.seed(seed)
    np.random.shuffle(x)
    np.random.seed(seed)
    np.random.shuffle(y)


random_list(train_x3, train_y3)
train_tf3 = tf.data.Dataset.from_tensor_slices((train_x3, train_y3)).batch(128)#x shape (batch,200)
valid_tf3 = tf.data.Dataset.from_tensors((valid_x3, valid_y3))

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
        tf.keras.layers.Dense(dff_, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model_)  # (batch_size, seq_len, d_model)
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


# class DecoderLayer(tf.keras.layers.Layer):
#     def __init__(self, d_model_, num_heads_, dff_, rate=0.1):
#         super(DecoderLayer, self).__init__()
#         self.mha1 = MultiHeadAttention(d_model_, num_heads_)
#         self.mha2 = MultiHeadAttention(d_model_, num_heads_)
#         self.ffn = point_wise_feed_forward_network(d_model_, dff_)
#         self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         self.dropout1 = tf.keras.layers.Dropout(rate)
#         self.dropout2 = tf.keras.layers.Dropout(rate)
#         self.dropout3 = tf.keras.layers.Dropout(rate)
#
#     def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
#         # enc_output.shape == (batch_size, input_seq_len, d_model)
#         attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
#         attn1 = self.dropout1(attn1, training=training)
#         out1 = self.layernorm1(attn1 + x)
#         attn2, attn_weights_block2 = self.mha2(
#             enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
#         attn2 = self.dropout2(attn2, training=training)
#         out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
#         ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
#         ffn_output = self.dropout3(ffn_output, training=training)
#         out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
#         return out3, attn_weights_block1, attn_weights_block2
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


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers_, d_model_, num_heads_, dff_, maximum_position_encoding, seq_len, rate=0.1): #input_vocab and max_vocab are the same
        super(Encoder, self).__init__()
        self.d_model = d_model_
        self.num_layers = num_layers_
        self.embedding = tf.keras.layers.Embedding(maximum_position_encoding, d_model_)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        self.enc_layers = [EncoderLayer(d_model_, num_heads_, dff_, rate) for _ in range(num_layers_)]
        #self.enc_layers = EncoderLayer(d_model_, num_heads_, dff_, rate)
        self.dropout = tf.keras.layers.Dropout(rate)
        #self.semi_final = tf.keras.layers.RNN(tf.keras.layers.GRUCell(args.n_hid))
        self.semi_final = tf.keras.layers.MaxPool1D([seq_len])
        self.final_layer = tf.keras.layers.Dense(args.n_out, activation='sigmoid')

    def call(self, x_, training, mask_att, justmask):
        seq_len = tf.shape(x_)[1]
        # adding embedding and position encoding.
        x_ = self.embedding(x_)  # (batch_size, input_seq_len, d_model)
        x_ *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x_ += self.pos_encoding[:, :seq_len, :]
        x_ = self.dropout(x_, training=training)
        for i in range(self.num_layers):
            x_ = self.enc_layers[i](x_, training, mask_att)
        # x shape (batch_size, input_seq_len, d_model)
        #x_ = tf.keras.layers.Reshape((-1, self.d_model))(x_)
        out = self.semi_final(x_)  # (batch_size, 256)
        out = self.final_layer(out) # (batch_size, 1)
        return tf.squeeze(out)


# class Decoder(tf.keras.layers.Layer):
#     def __init__(self, num_layers_, d_model_, num_heads_, dff_, target_vocab_size_, maximum_position_encoding, rate=0.1):
#         super(Decoder, self).__init__()
#         self.d_model = d_model_
#         self.num_layers = num_layers_
#         self.embedding = tf.keras.layers.Embedding(target_vocab_size_, d_model_)
#         self.pos_encoding = positional_encoding(maximum_position_encoding, d_model_)
#         self.dec_layers = [DecoderLayer(d_model_, num_heads_, dff_, rate) for _ in range(num_layers)]
#         self.dropout = tf.keras.layers.Dropout(rate)
#
#     def call(self, x_, enc_output, training, look_ahead_mask, padding_mask):
#         seq_len = tf.shape(x_)[1]
#         attention_weights = {}
#         x_ = self.embedding(x_)  # (batch_size, target_seq_len, d_model)
#         x_ *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
#         x_ += self.pos_encoding[:, :seq_len, :]
#         x_ = self.dropout(x_, training=training)
#         for i in range(self.num_layers):
#             x_, block1, block2 = self.dec_layers[i](x_, enc_output, training, look_ahead_mask, padding_mask)
#             attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
#             attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2
#         # x.shape == (batch_size, target_seq_len, d_model)
#         return x_, attention_weights
# class Transformer(tf.keras.Model):
#     def __init__(self, num_layers_, d_model_, num_heads_, dff_, input_vocab_size_,
#                  target_vocab_size, pe_input, pe_target, rate=0.1):
#         super(Transformer, self).__init__()
#         self.encoder = Encoder(num_layers_, d_model_, num_heads_, dff_, input_vocab_size_, pe_input, rate)
#         self.decoder = Decoder(num_layers_, d_model_, num_heads_, dff_, target_vocab_size, pe_target, rate)
#         self.final_layer = tf.keras.layers.Dense(target_vocab_size)
#
#     def call(self, inp_, tar_, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
#         enc_output = self.encoder(inp_, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
#         # dec_output.shape == (batch_size, tar_seq_len, d_model)
#         dec_output, attention_weights = self.decoder(tar_, enc_output, training, look_ahead_mask, dec_padding_mask)
#         final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
#         return final_output, attention_weights


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


#learning_rate = CustomSchedule(args.d_model)
optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')


def loss_function(real, pred):
    #mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    #mask = tf.cast(mask, dtype=loss_.dtype)
    #loss_ *= mask
    return tf.reduce_mean(loss_)


train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
AUCFunc = tf.keras.metrics.AUC()
accFunc = tf.keras.metrics.BinaryAccuracy()
precFunc = tf.keras.metrics.Precision(name='precFunc')
recallFunc = tf.keras.metrics.Recall(name='recallFunc')
encoder = Encoder(args.num_layers, args.d_model, args.num_heads, args.dff, args.max_vocab_size, seq_len=args.seq_size, rate=args.dropout_rate)
# transformer = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size,
#                           pe_input=input_vocab_size, pe_target=target_vocab_size, rate=dropout_rate)
#def create_look_ahead_mask(size):
#    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
#    return mask  # (seq_len, seq_len)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :], seq# (batch_size, 1, 1, seq_len)

"""
def create_masks(inp_):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp_)
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp_)
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar_)[1])
    dec_target_padding_mask = create_padding_mask(tar_)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask
"""

# checkpoint_path = "./checkpoints/train"
# ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# # if a checkpoint exists, restore the latest checkpoint.
# if ckpt_manager.latest_checkpoint:
#     ckpt.restore(ckpt_manager.latest_checkpoint)
#     print('Latest checkpoint restored!!')

# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]


@tf.function(input_signature=train_step_signature)
def train_step(inp_, real):  # shape is [batch, seq_len]
    #enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp)
    inp_padding_mask, justmask = create_padding_mask(inp_)
    with tf.GradientTape() as tape:
        #predictions, _ = transformer(inp_, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
        pred = encoder(inp_, True, inp_padding_mask, justmask)
        loss = loss_function(real, pred)
    gradients = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))
    train_loss(loss)
    precFunc.update_state(y_true=real, y_pred=pred)
    recallFunc.update_state(y_true=real, y_pred=pred)
    AUCFunc.update_state(y_true=real, y_pred=pred)
    accFunc.update_state(y_true=real, y_pred=pred)


for epoch in range(args.epochs):
    start = time.time()
    train_loss.reset_states()
    precFunc.reset_states()
    recallFunc.reset_states()
    AUCFunc.reset_states()
    accFunc.reset_states()
    # inp -> portuguese, tar -> english
    for (batch, (X, Y)) in enumerate(train_tf3):
        train_step(X, Y)
    # if (epoch + 1) % 5 == 0:
    #     ckpt_save_path = ckpt_manager.save()
    #     print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
    print('Epoch {} Loss {:.4f}'.format(epoch + 1, train_loss.result()))
    print('prec {:.4f} recall {:.4f} AUC {:.4f}, acc {:.4f}'.format(precFunc.result(), recallFunc.result(),
                                                                    AUCFunc.result(), accFunc.result()))
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

# def evaluate(inp_sentence):
#     start_token = [tokenizer_pt.vocab_size]
#     end_token = [tokenizer_pt.vocab_size + 1]
#     # inp sentence is portuguese, hence adding the start and end token
#     inp_sentence = start_token + tokenizer_pt.encode(inp_sentence) + end_token
#     encoder_input = tf.expand_dims(inp_sentence, 0)
#     # as the target is english, the first word to the transformer should be the
#     # english start token.
#     decoder_input = [tokenizer_en.vocab_size]
#     output = tf.expand_dims(decoder_input, 0)
#     for i in range(MAX_LENGTH):
#         enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
#         # predictions.shape == (batch_size, seq_len, vocab_size)
#         predictions, attention_weights = transformer(encoder_input, output, False, enc_padding_mask, combined_mask,
#                                                      dec_padding_mask)
#         # select the last word from the seq_len dimension
#         predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
#         predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
#         # return the result if the predicted_id is equal to the end token
#         if predicted_id == tokenizer_en.vocab_size + 1:
#             return tf.squeeze(output, axis=0), attention_weights
#         # concatentate the predicted_id to the output which is given to the decoder
#         # as its input.
#         output = tf.concat([output, predicted_id], axis=-1)
#     return tf.squeeze(output, axis=0), attention_weights
