import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from numpy import mean


def cross_valididation(session, model, x, y, batch_size=64, epochs=5, K=5, verbose=True, print_interval=100):
    # https://stackoverflow.com/questions/39748660/how-to-perform-k-fold-cross-validation-with-tensorflow
    loss_op, accuracy_op, train_op = model

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

        for e in range(epochs):
            # Training
            # (Note: X, Y, and training are used implicitly by loss_op, accuracy_op, and train_op. Do not remove them.)
            sum_loss, sum_accuracy = 0, 0
            training = tf.constant(True)
            for b in range(num_train_batches):
                X, Y = tf.train.batch([x[train_i], y[train_i]], batch_size=batch_size, allow_smaller_final_batch=False)
                loss, accuracy, _ = session.run([loss_op, accuracy_op, train_op])

                if verbose and b % print_interval == 0:
                    print("Batch {}: Loss = {}, Accuracy = {}".format(b, loss, accuracy))
                sum_loss += loss
                sum_accuracy += accuracy

            train_loss[k].append(sum_loss / num_train_batches)
            train_accuracy[k].append(sum_accuracy / num_train_batches)
            if verbose:
                print("Epoch {}: Average Train Loss = {}, Average Train Accuracy = {}\n"
                      .format(e + 1, train_loss[k][e], train_accuracy[k][e]))

        # Validation
        # (Note: X, Y, and training are used implicitly by loss_op, accuracy_op, and train_op. Do not remove them.)
        sum_loss, sum_accuracy = 0, 0
        training = tf.constant(False)
        for b in range(num_valid_batches):
            X, Y = tf.train.batch([x[valid_i], y[valid_i]], batch_size=batch_size, allow_smaller_final_batch=False)
            loss, accuracy = session.run([loss_op, accuracy_op])
            sum_loss += loss
            sum_accuracy += accuracy

        valid_loss.append(sum_loss / num_valid_batches)
        valid_accuracy.append(sum_accuracy / num_valid_batches)
        if verbose:
            print("Fold {}: Validation Loss = {}, Validation Accuracy = {}"
                  .format(k + 1, valid_loss[k], valid_accuracy[k]))

        k += 1

    print("Average Valid Loss = {}, Average Valid Accuracy = {}".format(mean(valid_loss), mean(valid_accuracy)))

    plt.figure()
    plt.title("Training Loss per Epoch")
    plt.plot(np.arange(epochs), np.array(train_loss))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Fold %d" % i for i in range(1, K+1)])
    plt.show()


def save(session, save_path):
    saver = tf.train.Saver()
    saver.save(session, save_path)


def load(session, save_path):
    saver = tf.train.Saver()
    saver.restore(session, save_path)


