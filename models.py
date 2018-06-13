import tensorflow as tf
import numpy as np


def model1(image_size, image_history_length):
    X = tf.placeholder(tf.float32, [None, image_history_length, *image_size])
    Y = tf.placeholder(tf.int64, [None])
    training = tf.placeholder(tf.bool)

    with tf.variable_scope("model1"):
        with tf.variable_scope("Model"):
            conv1 = _conv_axis1_loop(X, filters=4, kernel_size=[7, 7], strides=[4, 4],
                                     padding="valid", activation=tf.nn.relu,
                                     name="conv1", reuse=tf.AUTO_REUSE)
            conv2 = _conv_axis1_loop(conv1, filters=4, kernel_size=[7, 7], strides=[4, 4],
                                     padding="valid", activation=tf.nn.relu,
                                     name="conv2", reuse=tf.AUTO_REUSE)
            flat2 = tf.layers.flatten(conv2)
            dens4 = tf.layers.dense(flat2, units=64)
            dens5 = tf.layers.dense(dens4, units=16)
            logits = tf.layers.dense(dens5, units=2)

        with tf.variable_scope("Predict"):
            predict_op = tf.nn.softmax(logits)

        with tf.variable_scope("Loss"):
            loss_op = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=tf.one_hot(Y, 2))

        with tf.variable_scope("Accuracy"):
            prediction = tf.argmax(logits, axis=1)
            accuracy_op = tf.reduce_mean(tf.cast(tf.equal(prediction, Y), tf.float32))

        with tf.variable_scope("Optimizer"):
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(loss_op)  # TODO: tf.cond(): return None if not training

    return [predict_op, loss_op, accuracy_op, train_op], (X, Y, training)


def model2(image_size):
    X = tf.placeholder(tf.float32, [None, *image_size])
    Y = tf.placeholder(tf.int64, [None])
    training = tf.placeholder(tf.bool)

    with tf.variable_scope("Model2"):
        with tf.variable_scope("Model"):
            conv1 = tf.layers.conv2d(X, filters=16, kernel_size=[7, 7], strides=[2, 2],
                                     padding="same", activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(conv1, filters=16, kernel_size=[7, 7], strides=[2, 2],
                                     padding="same", activation=tf.nn.relu)
            maxp1 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=[1, 1], padding="valid")

            conv3 = tf.layers.conv2d(maxp1, filters=16, kernel_size=[7, 7], strides=[2, 2],
                                     padding="same", activation=tf.nn.relu)
            conv4 = tf.layers.conv2d(conv3, filters=16, kernel_size=[7, 7], strides=[2, 2],
                                     padding="same", activation=tf.nn.relu)
            maxp2 = tf.layers.max_pooling2d(conv4, pool_size=[2, 2], strides=[1, 1], padding="valid")

            flat1 = tf.layers.flatten(maxp2)
            dens1 = tf.layers.dense(flat1, units=512, activation=tf.nn.relu)
            dens2 = tf.layers.dense(dens1, units=64)
            logits = tf.layers.dense(dens2, units=3)

        with tf.variable_scope("Predict"):
            predict_op = tf.nn.softmax(logits)

        with tf.variable_scope("Loss"):
            loss_op = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=tf.one_hot(Y, 3))

        with tf.variable_scope("Accuracy"):
            prediction = tf.argmax(logits, axis=1)
            accuracy_op = tf.reduce_mean(tf.cast(tf.equal(prediction, Y), tf.float32))

        with tf.variable_scope("Optimizer"):
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(loss_op)  # TODO: tf.cond(): return None if not training

    return [predict_op, loss_op, accuracy_op, train_op], (X, Y, training)


def _conv_axis1_loop(X, filters, kernel_size, strides, padding, activation, name, reuse):
    new_shape = [-1] + [d for i, d in enumerate(X.shape[1:]) if i != 0]
    conc = []
    for i in range(X.shape[1]):
        x = tf.reshape(X[:, i], new_shape)
        l = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                             activation=activation, name=name, reuse=reuse)
        conc.append(tf.expand_dims(l, 1))
    return tf.concat(conc, axis=1)