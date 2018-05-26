import tensorflow as tf
import os

import train


# https://blog.metaflow.fr/tensorflow-how-to-optimise-your-input-pipeline-with-queues-and-multi-threading-e7c3874157e0


def model(image_size):
    X = tf.placeholder(tf.float32, [None, *image_size])
    Y = tf.placeholder(tf.int32, [None])
    training = tf.placeholder(tf.bool)

    with tf.variable_scope("Model"):
        conv1 = tf.layers.conv2d(X, filters=16, kernel_size=[9,9], strides=[4,4],
                                 padding="valid", activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, filters=16, kernel_size=[9,9], strides=[4,4],
                                 padding="valid", activation=tf.nn.relu)
        maxp1 = tf.layers.max_pooling2d(conv2, pool_size=[4,4], strides=[3,3], padding="valid")

        conv3 = tf.layers.conv2d(maxp1, filters=16, kernel_size=[9, 9], strides=[4, 4],
                                 padding="valid", activation=tf.nn.relu)
        conv4 = tf.layers.conv2d(conv3, filters=16, kernel_size=[9, 9], strides=[4, 4],
                                 padding="valid", activation=tf.nn.relu)
        maxp2 = tf.layers.max_pooling2d(conv4, pool_size=[4, 4], strides=[3, 3], padding="valid")

        flat1 = tf.layers.flatten(maxp2)
        dens1 = tf.layers.dense(flat1, units=512, activation=tf.nn.relu)
        dens2 = tf.layers.dense(dens1, units=64)
        logits = tf.layers.dense(dens2, units=3)

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


if __name__ == "__main__":
    save_path = "/tmp/model2.ckpt"

    if os.path.exists(save_path):
        r = input("Saved Data already exists for Model2. Do you want to overwrite this? (y/n): ")
        if not r.lower().startswith("y"):
            exit()

    with tf.Session() as session:
        # TODO: Get Data, Test, ...
        # X, Y, m = None, None, model([32, 32, 3])
        # train.cross_valididation(session, m, X, Y)
        # train.save(session, save_path)

        raise NotImplementedError("Not Yet Implemented")

