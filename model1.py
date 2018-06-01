import tensorflow as tf
import numpy as np
import os

import train
import loadDataset


def _conv_axis1_loop(X, filters, kernel_size, strides, padding, activation, name, reuse):
    new_shape = [-1] + [d for i, d in enumerate(X.shape[1:]) if i != 0]
    conc = []
    for i in range(X.shape[1]):
        x = tf.reshape(X[:, i], new_shape)
        l = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                             activation=activation, name=name, reuse=reuse)
        conc.append(tf.expand_dims(l, 1))
    return tf.concat(conc, axis=1)


# https://blog.metaflow.fr/tensorflow-how-to-optimise-your-input-pipeline-with-queues-and-multi-threading-e7c3874157e0

def construct(image_size):
    X = tf.placeholder(tf.float32, [None, 45, *image_size])
    Y = tf.placeholder(tf.int64, [None])
    training = tf.placeholder(tf.bool)

    with tf.variable_scope("model1"):
        with tf.variable_scope("Model"):
            conv1 = _conv_axis1_loop(X, filters=1, kernel_size=[7,7], strides=[4,4],
                                     padding="valid", activation=tf.nn.relu,
                                     name="conv1", reuse=tf.AUTO_REUSE)
            conv2 = _conv_axis1_loop(conv1, filters=1, kernel_size=[7,7], strides=[4,4],
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


def split(X, k):
    n = X.shape[0]
    for i in range(n - k):
        yield X[i:(i+k)]


if __name__ == "__main__":
    save_path = "/tmp/model1.ckpt"

    if os.path.exists(save_path):
        r = input("Saved Data already exists for Model1. Do you want to overwrite this? (y/n): ")
        if not r.lower().startswith("y"):
            exit()

    with tf.Session() as session:
        print("Loading Data...")
        X, Y = loadDataset.loadDataSet('./dataset/imgs/rock_frames', './dataset/imgs/paper_frames',
                                       './dataset/imgs/scissor_frames', './dataset/csvs/rock.csv',
                                       './dataset/csvs/paper.csv', './dataset/csvs/scissor.csv')
        Y = (Y > 0).astype(int)  # 0: None/NA, 1: Rock/Paper/Scissors

        X = (X.astype(np.float32) - 128) / 128  # Normalize

        X = np.concatenate([a[None] for a in split(X, 45)])  # (N-45) x 45 x H x W x 3
        Y = Y[-X.shape[0]:]  # (N-45) x 1
        X, Y = train.shuffle([X, Y])

        print("Constructing Model...")
        model = construct(X.shape[2:])

        print("Training...")
        train.cross_valididation(session, model, X, Y, epochs=5, print_interval=10)