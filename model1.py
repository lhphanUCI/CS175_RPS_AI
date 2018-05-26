import tensorflow as tf
import os

import train


# https://blog.metaflow.fr/tensorflow-how-to-optimise-your-input-pipeline-with-queues-and-multi-threading-e7c3874157e0


def model(image_size):
    X = tf.placeholder(tf.float32, [None, *image_size])
    Y = tf.placeholder(tf.int32, [None])
    training = tf.placeholder(tf.bool)

    with tf.variable_scope("Model"):
        conv1 = tf.map_fn(lambda x: tf.layers.conv2d(x, filters=1, kernel_size=[9,9], strides=[4,4],
                                                     padding="valid", activation=tf.nn.relu,
                                                     name="conv1", reuse=True), X)
        conv2 = tf.map_fn(lambda x: tf.layers.conv2d(x, filters=1, kernel_size=[9,9], strides=[4,4],
                                                     padding="valid", activation=tf.nn.relu,
                                                     name="conv2", reuse=True), conv1)
        maxp1 = tf.map_fn(lambda x: tf.layers.max_pooling2d(x, pool_size=[4,4], strides=[3,3], padding="valid"), conv2)
        flat1 = tf.map_fn(lambda x: tf.layers.flatten(x), maxp1)
        dens1 = tf.map_fn(lambda x: tf.layers.dense(x, units=64, activation=tf.nn.relu,
                                                    name="dens1", reuse=True), flat1)
        dens2 = tf.map_fn(lambda x: tf.layers.dense(x, units=1, activation=tf.nn.relu,
                                                    name="dens2", reuse=True), dens1)
        flat2 = tf.layers.flatten(dens2)
        dens3 = tf.layers.dense(flat2, units=64)  # Consider changing the number of units.
        logits = tf.layers.dense(dens3, units=2)

    with tf.variable_scope("Predict"):
        predict_op = tf.nn.softmax(logits)

    with tf.variable_scope("Loss"):
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y)
        loss_op = tf.reduce_mean(losses)

    with tf.variable_scope("Accuracy"):
        prediction = tf.argmax(logits, axis=1)
        accuracy_op = tf.reduce_mean(tf.equal(prediction, Y))

    with tf.variable_scope("Optimizer"):
        optimizer = tf.train.AdamOptimizer()
        train_op = tf.cond(training, optimizer.minimize(loss_op), None)

    return [predict_op, loss_op, accuracy_op, train_op], (X, Y, training)


def keras_model():
    pass


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

