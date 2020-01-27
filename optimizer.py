import tensorflow as tf

tf.compat.v1.disable_v2_behavior()

class GraphGANOptimizer(object):
    def __init__(self, model, learning_rate=1e-3, feature_matching=True):
        self.la = tf.placeholder_with_default(1., shape=())
        with tf.name_scope('losses'):
            eps = tf.random_uniform(tf.shape(model.features_real)[:1], dtype=model.features_real.dtype)
            x_int0 = model.input2gen * tf.expand_dims(tf.expand_dims(eps, -1), -1) + model.logit_softmax *\
                     (1 - tf.expand_dims(tf.expand_dims(eps, -1), -1))
            grad0 = tf.gradients(model.D_x(x_int0, model.discriminator_units), x_int0)
            self.grad_penalty = tf.reduce_mean(((1 - tf.norm(grad0, axis=-1)) ** 2), (-2, -1))
            self.loss_D = - model.features_real + model.features_fake
            self.loss_G = - model.features_fake
            # self.loss_V = (model.value_real - model.rewardR) ** 2 + (model.value_fake - model.rewardF) ** 2
            self.loss_RL = - model.value_fake
            self.loss_F = (tf.reduce_mean(model.feature_logit_real, 0) - tf.reduce_mean(model.feature_logit_fake, 0)) ** 2

        self.loss_D = tf.reduce_mean(self.loss_D)
        self.loss_G = tf.reduce_sum(self.loss_F) if feature_matching else tf.reduce_mean(self.loss_G)
        # self.loss_V = tf.reduce_mean(self.loss_V)
        self.loss_RL = tf.reduce_mean(self.loss_RL)
        alpha = tf.abs(tf.stop_gradient(self.loss_G / self.loss_RL))
        self.grad_penalty = tf.reduce_mean(self.grad_penalty)

        with tf.name_scope('train_step'):
            self.train_step_D = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss=self.loss_D + 10 * self.grad_penalty,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))

            self.train_step_G = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss=tf.cond(tf.greater(self.la, 0), lambda: self.la * self.loss_G, lambda: 0.) + tf.cond(
                    tf.less(self.la, 1), lambda: (1 - self.la) * alpha * self.loss_RL, lambda: 0.),
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))

            # self.train_step_V = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
            #     loss=self.loss_V,
            #     var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='value'))
