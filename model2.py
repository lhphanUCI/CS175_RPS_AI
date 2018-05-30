import tensorflow as tf
import os

import train
import loadDataset


# https://blog.metaflow.fr/tensorflow-how-to-optimise-your-input-pipeline-with-queues-and-multi-threading-e7c3874157e0


def construct(image_size):
    X = tf.placeholder(tf.float32, [None, *image_size])
    Y = tf.placeholder(tf.int64, [None])
    training = tf.placeholder(tf.bool)

    with tf.variable_scope("Model"):
        conv1 = tf.layers.conv2d(X, filters=16, kernel_size=[7,7], strides=[2,2],
                                 padding="same", activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, filters=16, kernel_size=[7,7], strides=[2,2],
                                 padding="same", activation=tf.nn.relu)
        maxp1 = tf.layers.max_pooling2d(conv2, pool_size=[2,2], strides=[1,1], padding="valid")

        conv3 = tf.layers.conv2d(maxp1, filters=16, kernel_size=[7,7], strides=[2,2],
                                 padding="same", activation=tf.nn.relu)
        conv4 = tf.layers.conv2d(conv3, filters=16, kernel_size=[7,7], strides=[2,2],
                                 padding="same", activation=tf.nn.relu)
        maxp2 = tf.layers.max_pooling2d(conv4, pool_size=[2,2], strides=[1,1], padding="valid")

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


if __name__ == "__main__":
    save_path = "/tmp/model2.ckpt"

    if os.path.exists(save_path):
        r = input("Saved Data already exists for Model2. Do you want to overwrite this? (y/n): ")
        if not r.lower().startswith("y"):
            exit()

    with tf.Session() as session:
        print("Loading Data...")
        X, Y = loadDataset.loadDataSet('./dataset/imgs/rock_frames', './dataset/imgs/paper_frames',
                                       './dataset/imgs/scissor_frames', './dataset/csvs/rock.csv',
                                       './dataset/csvs/paper.csv', './dataset/csvs/scissor.csv')
        X, Y = X[Y > 0], Y[Y > 0] - 1  # Y is 0-based containing only Rock, Paper, or Scissors.
        X, Y = train.shuffle([X, Y])   # Order is irrelevant. Shuffle for better training.

        print("Constructing Model...")
        model = construct(X.shape[1:])

        print("Training...")
        train.cross_valididation(session, model, X, Y, epochs=5)


