import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from numpy import mean, argmax


def cross_valididation(session, model, x, y, batch_size=64, epochs=5, K=5, verbose=True, print_interval=100):
    # https://stackoverflow.com/questions/39748660/how-to-perform-k-fold-cross-validation-with-tensorflow
    [predict_op, loss_op, accuracy_op, train_op], (X, Y, training) = model

    # K-Fold Loop
    train_loss, train_accuracy = [], []
    valid_loss, valid_accuracy = [], []
    k = 0
    for train_i, valid_i in KFold(n_splits=K).split(x):
        train_loss.append([])
        train_accuracy.append([])

        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        num_train_batches = int(x[train_i].shape[0] / batch_size)
        num_valid_batches = int(x[valid_i].shape[0] / batch_size)

        # Training
        for e in range(epochs):
            sum_loss, sum_accuracy = 0, 0
            for batch_i, (batch_x, batch_y) in enumerate(_batches([x[train_i], y[train_i]], batch_size=batch_size, allow_smaller_final_batch=False)):
                loss, accuracy, _ = session.run([loss_op, accuracy_op, train_op],
                                                feed_dict={X: batch_x, Y: batch_y, training: True})

                if verbose and batch_i % print_interval == 0:
                    print("Train Batch {}: Loss = {}, Accuracy = {}".format(batch_i, loss, accuracy))
                sum_loss += loss
                sum_accuracy += accuracy

            train_loss[k].append(sum_loss / num_train_batches)
            train_accuracy[k].append(sum_accuracy / num_train_batches)
            if verbose:
                print("Epoch {}: Average Train Loss = {}, Average Train Accuracy = {}\n"
                      .format(e + 1, train_loss[k][e], train_accuracy[k][e]))

        # Validation
        sum_loss, sum_accuracy = 0, 0
        for batch_i, (batch_x, batch_y) in enumerate(_batches([x[valid_i], y[valid_i]], batch_size=batch_size, allow_smaller_final_batch=False)):
            loss, accuracy = session.run([loss_op, accuracy_op],
                                         feed_dict={X: batch_x, Y: batch_y, training: False})

            if verbose and batch_i % print_interval == 0:
                print("Valid Batch {}: Loss = {}, Accuracy = {}".format(batch_i, loss, accuracy))
            sum_loss += loss
            sum_accuracy += accuracy

        valid_loss.append(sum_loss / num_valid_batches)
        valid_accuracy.append(sum_accuracy / num_valid_batches)
        if verbose:
            print("Fold {}: Validation Loss = {}, Validation Accuracy = {}\n"
                  .format(k + 1, valid_loss[k], valid_accuracy[k]))

        k += 1

    # Results
    print("Average Valid Loss = {}, Average Valid Accuracy = {}".format(mean(valid_loss), mean(valid_accuracy)))

    plt.figure()
    plt.title("Training Loss per Epoch")
    plt.plot(np.arange(epochs), np.array(train_loss))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Fold %d" % i for i in range(1, K+1)])
    plt.show()


def predict(session, model, X):
    predict_op = model[0]
    return argmax(session.run([predict_op]))


def save(session, save_path):
    saver = tf.train.Saver()
    saver.save(session, save_path)


def load(session, save_path):
    saver = tf.train.Saver()
    saver.restore(session, save_path)


def _batches(inputs, batch_size, shuffle=False, allow_smaller_final_batch=False):
    if not isinstance(inputs, list) or not isinstance(inputs, tuple):
        raise TypeError("Inputs must be of type list or tuple.")
    total_size = len(inputs[0])
    if not all([len(x) == total_size for x in inputs]):
        raise RuntimeError("All inputs must have equal first dimension.")

    order = np.arange(total_size) if shuffle is False else np.random.permutation(total_size)
    for i in range(int(total_size / batch_size)):
        yield [x[order[(i)*batch_size:(i+1)*batch_size]] for x in inputs]
    if allow_smaller_final_batch:
        yield [x[order[int(total_size / batch_size)*batch_size:]] for x in inputs]
