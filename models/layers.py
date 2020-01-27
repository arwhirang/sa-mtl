import tensorflow as tf

tf.compat.v1.disable_v2_behavior()


# discriminator
def encoder_rgcn(hidden_tensor, units, training, dropout_rate=0.):
    first_units, second_units = units
    with tf.variable_scope('graph_convolutions'):
        output = multi_dense_layers(hidden_tensor, first_units, training=training, activation=tf.nn.tanh,
                                    dropout_rate=dropout_rate)
    with tf.variable_scope('graph_aggregation'):
        annotations = tf.concat([output, hidden_tensor], -1)
        output = aggregation_layer(annotations, second_units, activation=tf.nn.tanh, dropout_rate=dropout_rate,
                                   training=training)
    return output


# generator
def decoder_adj(inputs, units, atomsize, lensize, training, dropout_rate=0.):
    output = multi_dense_layers(inputs, units, activation=tf.nn.tanh, dropout_rate=dropout_rate, training=training)
    with tf.variable_scope('nodes_logits'):
        logits = tf.layers.dense(inputs=output, units=atomsize * lensize, activation=None)
        logits = tf.reshape(logits, (-1, atomsize, lensize))
        logits = tf.layers.dropout(logits, dropout_rate, training=training)
    return logits


def aggregation_layer(inputs, units, training, activation=None, dropout_rate=0.):
    i = tf.layers.dense(inputs, units=units, activation=tf.nn.sigmoid)
    j = tf.layers.dense(inputs, units=units, activation=activation)
    output = tf.reduce_sum(i * j, 1)
    output = activation(output) if activation is not None else output
    output = tf.layers.dropout(output, dropout_rate, training=training)
    return output


def multi_dense_layers(inputs, units, training, activation=None, dropout_rate=0.):
    hidden_tensor = inputs
    for u in units:
        hidden_tensor = tf.layers.dense(hidden_tensor, units=u, activation=activation)
        hidden_tensor = tf.layers.dropout(hidden_tensor, dropout_rate, training=training)
    return hidden_tensor
