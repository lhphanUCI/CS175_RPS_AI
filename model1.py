import tensorflow as tf
import os

import train


# https://blog.metaflow.fr/tensorflow-how-to-optimise-your-input-pipeline-with-queues-and-multi-threading-e7c3874157e0


def model(image_size):
    X = tf.placeholder(tf.float32, [None, *image_size])
    Y = tf.placeholder(tf.int32, [None])
    training = tf.placeholder(tf.bool)

    with tf.variable_scope("Model"):
        # TODO: Implement Model Definition

        # Arbitrary Example
        l = X
        l = tf.layers.conv2d(l, filters=16, kernel_size=[5, 5], strides=[2, 2], padding="same", activation=tf.nn.relu)
        l = tf.layers.conv2d(l, filters=8, kernel_size=[5, 5], strides=[2, 2], padding="same", activation=tf.nn.relu)
        l = tf.layers.max_pooling2d(l, pool_size=[2, 2], strides=[2, 2])
        l = tf.layers.flatten(l)
        l = tf.layers.dense(l, units=100, activation=tf.nn.relu)
        l = tf.layers.dense(l, units=50, activation=tf.nn.relu)
        logits = tf.layers.dense(l, units=2)

        raise NotImplementedError("Not Yet Implemented")

    with tf.variable_scope("Loss"):
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y)
        loss_op = tf.reduce_mean(losses)

    with tf.variable_scope("Accuracy"):
        pred = tf.argmax(logits, axis=1)
        accuracy_op = tf.reduce_mean(tf.equal(pred, Y))

    with tf.variable_scope("Optimizer"):
        optimizer = tf.train.AdamOptimizer()
        train_op = tf.cond(training, optimizer.minimize(loss_op), None)

    return [loss_op, accuracy_op, train_op]


if __name__ == "__main__":
    save_path = "/tmp/model1.ckpt"

    if os.path.exists(save_path):
        r = input("Saved Data already exists for Model1. Do you want to overwrite this? (y/n): ")
        if not r.lower().startswith("y"):
            exit()

    with tf.Session() as session:
        # TODO: Get Data, Test, ...
        # X, Y, m = None, None, model([32, 32, 3])
        # train.cross_valididation(session, m, X, Y)
        # train.save(session, save_path)

        raise NotImplementedError("Not Yet Implemented")

