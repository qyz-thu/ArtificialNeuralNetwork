# -*- coding: utf-8 -*-

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS


class Model:
    def __init__(self,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9995):
        self.x_ = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 3])
        self.y_ = tf.placeholder(tf.int32, [FLAGS.batch_size])
        self.hidden1 = FLAGS.hidden_size1
        self.hidden2 = FLAGS.hidden_size2
        self.initializer = tf.contrib.layers.xavier_initializer()

        self.loss, self.pred, self.acc = self.forward(is_train=True, reuse=False)
        self.loss_val, self.pred_val, self.acc_val = self.forward(is_train=False, reuse=True)

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                    var_list=self.params)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def forward(self, is_train, reuse=None):
        with tf.variable_scope("model", reuse=reuse):
            x = tf.layers.conv2d(self.x_, filters=self.hidden1, kernel_size=[3, 3], strides=[1, 1], padding="same", kernel_initializer=self.initializer)
            x = batch_normalization_layer(x, is_train)
            x = tf.nn.relu(x)
            x = dropout_layer(x, FLAGS.drop_rate, is_train)
            x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2], padding='valid')
            x = tf.layers.conv2d(x, filters=self.hidden2, kernel_size=[3, 3], strides=[1, 1], padding='same', kernel_initializer=self.initializer)
            x = batch_normalization_layer(x, is_train)
            x = tf.nn.relu(x)
            x = dropout_layer(x, FLAGS.drop_rate, is_train)
            x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2], padding='valid')
            x = tf.reshape(x, shape=[-1, 8 * 8 * self.hidden2])
            logits = tf.layers.dense(x, units=10)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        pred = tf.argmax(logits, 1)  # Calculate the prediction result
        correct_pred = tf.equal(tf.cast(pred, tf.int32), self.y_)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # Calculate the accuracy in this mini-batch
        
        return loss, pred, acc


def batch_normalization_layer(incoming, is_train=True):
    return tf.layers.batch_normalization(inputs=incoming, training=is_train)


def dropout_layer(incoming, drop_rate, is_train=True):
    if is_train:
        return tf.nn.dropout(incoming, drop_rate)
    return incoming
