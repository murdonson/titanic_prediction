import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from util import dataGenerate
import pandas as pd

def Model(xtrain, ytrain, xtest, steps=1000, learning_rate=0.01, K=5, display_information=100, fm=True, seed=0):
    tf.set_random_seed(seed)
    n = xtrain.shape[1]

    X = tf.placeholder(tf.float32, shape=[None, n], name="X")
    y = tf.placeholder(tf.float32, shape=[None, 1], name="y")
    V = tf.get_variable("v", shape=[n, K], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.3))
    W = tf.get_variable("Weights", shape=[n, 1], dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.01))
    b = tf.get_variable("Biases", shape=[1, 1], dtype=tf.float32, initializer=tf.zeros_initializer())
    logits = tf.matmul(X, W) + b
    if fm:
        # FM 部分
        fm_hat = tf.reduce_sum(np.square(tf.matmul(X, V)) - (tf.matmul(tf.multiply(X, X), tf.multiply(V, V))), axis=1,
                               keep_dims=True) / 2
        logits = logits + fm_hat
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
    else:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))

    y_hat = tf.nn.sigmoid(logits)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(steps):
            loss_, _, y_h = sess.run([loss, optimizer, y_hat], feed_dict={X: xtrain, y: ytrain})
            if i % display_information == 0:
                print("Train accuracy is %.6f loss is %.6f" % (
                    accuracy_score(ytrain.reshape(-1, ), np.where(np.array(y_h).reshape(-1, ) >= 0.5, 1, 0)),
                    loss_))

        y_predict = sess.run(y_hat, feed_dict={X: xtest})
        predictions = np.where(np.array(y_predict).reshape(-1, ) >= 0.5, 1, 0)
        df = pd.read_csv("./Dataset/test.csv")
        ids = df["PassengerId"]
        output = pd.DataFrame({'PassengerId': ids, 'Survived': predictions})
        output.to_csv('titanic-predictions.csv', index=False)


if __name__ == "__main__":
    # dataTrain, labelTrain = loadDataSet("train_data.txt")
    x_train, y_train, field_dict = dataGenerate("train", path="./Dataset/train.csv")

    x_test = dataGenerate("test", path="./Dataset/test.csv")

    Model(x_train, y_train, x_test, 1000, 0.01, fm=True)
