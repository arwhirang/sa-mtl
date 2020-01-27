import numpy as np
import tensorflow as tf

from models.layers import multi_dense_layers

tf.compat.v1.disable_v2_behavior()

def postprocess_logits(inputs, temperature=1.):
    softmax = tf.nn.softmax(inputs / temperature)
    argmax = tf.one_hot(tf.argmax(inputs, axis=-1), depth=inputs.shape[-1], dtype=inputs.dtype)
    gumbel_logits = inputs - tf.log(- tf.log(tf.random_uniform(tf.shape(inputs), dtype=inputs.dtype)))
    gumbel_softmax = tf.nn.softmax(gumbel_logits / temperature)
    gumbel_argmax = tf.one_hot(tf.argmax(gumbel_logits, axis=-1), depth=gumbel_logits.shape[-1],
                               dtype=gumbel_logits.dtype)

    return [softmax, argmax, gumbel_logits, gumbel_softmax, gumbel_argmax]


class GraphGANModel(object):
    def __init__(self, atomsize, lensize, embedding_dim, decoder_units, discriminator_units,
                 decoder, discriminator, batch_discriminator=True):
        self.embedding_dim, self.decoder_units, self.discriminator_units, self.decoder, self.discriminator, \
        self.batch_discriminator = embedding_dim, decoder_units, discriminator_units, decoder, discriminator, \
                                   batch_discriminator
        soft_gumbel_softmax = False
        hard_gumbel_softmax = False

        self.training = tf.placeholder(dtype=tf.bool, shape=())
        self.dropout_rate = tf.placeholder(dtype=tf.float, shape=())

        self.soft_gumbel_softmax = tf.placeholder(soft_gumbel_softmax, shape=())
        self.hard_gumbel_softmax = tf.placeholder(hard_gumbel_softmax, shape=())
        self.temperature = tf.placeholder(1., shape=())

        self.input2gen = tf.placeholder(dtype=tf.int64, shape=(None, atomsize, lensize))
        self.embeddings = tf.placeholder(dtype=tf.float32, shape=(None, embedding_dim))#initial noise
        # self.rewardR = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        # self.rewardF = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        self.atomsize = atomsize
        self.lensize = lensize
        with tf.variable_scope('generator'):  # shape (-1, atomsize, lensize)
            self.logits = self.decoder(self.embeddings, decoder_units, atomsize, lensize, training=self.training,
                                       dropout_rate=self.dropout_rate)

        with tf.name_scope('outputs'):
            self.logit_softmax, self.logit_argmax, self.logit_gumbel_logits, self.logit_gumbel_softmax, \
            self.logit_gumbel_argmax = postprocess_logits(self.logits, temperature=self.temperature)
            self.logit_hat = tf.case({self.soft_gumbel_softmax: lambda: self.logit_gumbel_softmax,
                                      self.hard_gumbel_softmax: lambda: tf.stop_gradient(self.logit_gumbel_argmax -
                                                                                         self.logit_gumbel_softmax) +
                                                                        self.logit_gumbel_softmax},
                                     default=lambda: self.logit_argmax, exclusive=True)

        with tf.name_scope('D_x_real'):
            self.features_real, self.feature_logit_real = self.D_x(self.input2gen, units=discriminator_units)
        with tf.name_scope('D_x_fake'):
            self.features_fake, self.feature_logit_fake = self.D_x(self.logit_hat, units=discriminator_units)

        with tf.name_scope('V_x_real'):
            self.value_real = self.V_x(self.input2gen, units=discriminator_units)
        with tf.name_scope('V_x_fake'):
            self.value_fake = self.V_x(self.logit_hat, units=discriminator_units)

    def D_x(self, inputs, units):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            outputs0 = self.discriminator(inputs, units=units[:-1], training=self.training,
                                          dropout_rate=self.dropout_rate)
            outputs1 = multi_dense_layers(outputs0, units=units[-1], activation=tf.nn.tanh, training=self.training,
                                          dropout_rate=self.dropout_rate)
            # if self.batch_discriminator:
            #     outputs_batch = tf.layers.dense(outputs0, units[-2] // 8, activation=tf.tanh)
            #     outputs_batch = tf.layers.dense(tf.reduce_mean(outputs_batch, 0, keep_dims=True), units[-2] // 8,
            #                                     activation=tf.nn.tanh)
            #     outputs_batch = tf.tile(outputs_batch, (tf.shape(outputs0)[0], 1))
            #
            #     outputs1 = tf.concat((outputs1, outputs_batch), -1)
            outputs = tf.layers.dense(outputs1, units=1)
        return outputs, outputs1

    def V_x(self, inputs, units):
        with tf.variable_scope('value', reuse=tf.AUTO_REUSE):
            outputs = self.discriminator(inputs, units=units[:-1], training=self.training,
                                         dropout_rate=self.dropout_rate)
            outputs = multi_dense_layers(outputs, units=units[-1], activation=tf.nn.tanh, training=self.training,
                                         dropout_rate=self.dropout_rate)
            outputs = tf.layers.dense(outputs, units=1, activation=tf.nn.sigmoid)
        return outputs

    def sample_z(self, batch_dim):
        return np.random.normal(0, 1, size=(batch_dim, self.embedding_dim))
