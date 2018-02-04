import tensorflow as tf
import numpy as np


class MLP(object):

    def __init__(
      self, sequence_length, num_classes, num_neurons, vocab_size,
      embedding_size, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout 
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (set to 0, not use regularization)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            #self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        with tf.name_scope("MLP"):
            means = tf.reduce_mean(self.embedded_chars,1)
            W = tf.Variable(tf.truncated_normal([embedding_size,num_neurons], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_neurons]), name="b")
            h = tf.add(tf.matmul(means, W),b)
            self.h_=tf.nn.relu(h,name='relu')


        with tf.name_scope("dropout"):
            self.h_ = tf.nn.dropout(self.h_, self.dropout_keep_prob)


        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_neurons,num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")