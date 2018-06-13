import tensorflow as tf
import numpy as np


def model1(image_size, image_history_length):
    X = tf.placeholder(tf.float32, [None, image_history_length, *image_size])
    Y = tf.placeholder(tf.int64, [None])
    training = tf.placeholder(tf.bool)

    with tf.variable_scope("model1"):
        with tf.variable_scope("Model"):
            conv1 = _conv_axis1_loop(X, filters=8, kernel_size=[7, 7], strides=[4, 4],
                                     padding="valid", activation=tf.nn.relu,
                                     name="conv1", reuse=tf.AUTO_REUSE)
            conv2 = _conv_axis1_loop(conv1, filters=4, kernel_size=[7, 7], strides=[4, 4],
                                     padding="valid", activation=tf.nn.relu,
                                     name="conv2", reuse=tf.AUTO_REUSE)

            flat1 = tf.layers.flatten(conv2)
            bnrm1 = tf.layers.batch_normalization(flat1)
            drop1 = tf.layers.dropout(bnrm1, rate=0.25)

            dens1 = tf.layers.dense(drop1, units=1024, activation=tf.nn.relu)
            dens2 = tf.layers.dense(dens1, units=256, activation=tf.nn.relu)
            dens3 = tf.layers.dense(dens2, units=64, activation=tf.nn.relu)
            logits = tf.layers.dense(dens3, units=2)

        with tf.variable_scope("Predict"):
            predict_op = tf.nn.softmax(logits)

        with tf.variable_scope("Loss"):
            class_weights = tf.gather(tf.constant([1, 1]), Y)  # Incorrect "0" vs "1" has different costs.
            loss_op = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=tf.one_hot(Y, 2), weights=class_weights)

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
                                     padding="same", activation=tf.nn.relu, name="conv1")
            conv2 = tf.layers.conv2d(conv1, filters=16, kernel_size=[7, 7], strides=[2, 2],
                                     padding="same", activation=tf.nn.relu, name="conv2")
            maxp1 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=[1, 1], padding="valid")

            conv3 = tf.layers.conv2d(maxp1, filters=16, kernel_size=[7, 7], strides=[2, 2],
                                     padding="same", activation=tf.nn.relu, name="conv3")
            conv4 = tf.layers.conv2d(conv3, filters=16, kernel_size=[7, 7], strides=[2, 2],
                                     padding="same", activation=tf.nn.relu, name="conv4")
            maxp2 = tf.layers.max_pooling2d(conv4, pool_size=[2, 2], strides=[1, 1], padding="valid")

            flat1 = tf.layers.flatten(maxp2)
            bnrm1 = tf.layers.batch_normalization(flat1)
            drop1 = tf.layers.dropout(bnrm1, rate=0.25)

            dens1 = tf.layers.dense(drop1, units=1024, activation=tf.nn.relu)
            dens2 = tf.layers.dense(dens1, units=256, activation=tf.nn.relu)
            dens3 = tf.layers.dense(dens2, units=64, activation=tf.nn.relu)
            logits = tf.layers.dense(dens3, units=3)

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


def model1_alt1(image_size, image_history_length):
    X = tf.placeholder(tf.float32, [64, image_history_length, *image_size])
    Y = tf.placeholder(tf.int64, [64])
    training = tf.placeholder(tf.bool)

    with tf.variable_scope("model1"):
        with tf.variable_scope("Model"):
            conv1 = _conv_axis1_loop(X, filters=8, kernel_size=[7, 7], strides=[2, 2],
                                     padding="valid", activation=tf.nn.relu,
                                     name="conv1", reuse=tf.AUTO_REUSE)
            conv2 = _conv_axis1_loop(conv1, filters=1, kernel_size=[7, 7], strides=[2, 2],
                                     padding="valid", activation=tf.nn.relu,
                                     name="conv2", reuse=tf.AUTO_REUSE)
            flat2 = tf.layers.flatten(conv2)

            lens = [(k, int(288 / k)) for k in [1, 2, 4, 6, 8]]
            conc = []
            for k, size in lens:
                sub_conc = []
                for j in range(k):
                    start, end = j * size, (j + 1) * size
                    mask = np.zeros((64, 288))
                    mask[start:end] = 1
                    l = tf.boolean_mask(flat2, mask)
                    l = tf.layers.dense(l, units=size / 4)
                    sub_conc.append(l)
                conc.append(tf.concat(sub_conc, axis=1))
            conc1 = tf.concat(conc, axis=1)  # Len = 5 * 288/4 = 360

            dens5 = tf.layers.dense(conc1, units=64)
            dens6 = tf.layers.dense(dens5, units=32)
            logits = tf.layers.dense(dens6, units=2)

        with tf.variable_scope("Predict"):
            predict_op = tf.nn.softmax(logits)

        with tf.variable_scope("Loss"):
            class_weights = tf.gather(tf.constant([1, 8]), Y)  # Incorrect "0" vs "1" has different costs.
            loss_op = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=tf.one_hot(Y, 2),
                                                      weights=class_weights)

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