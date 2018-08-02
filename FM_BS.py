# coding:utf-8
import tensorflow as tf
import numpy as np
import FileUtil
from sklearn.metrics import accuracy_score
import pandas as pd


class FM:
    def __init__(self, trainpath, testpath, k):
        self.x_train, self.y_train = FileUtil.readCSV("train", trainpath)
        self.x_test = FileUtil.readCSV("test", testpath)
        self.K = k

    def train(self):
        numberofFeatures = self.x_train.shape[1]
        # input
        X = tf.placeholder(tf.float32, shape=[None, numberofFeatures], name='X')
        # output
        y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
        # 一阶权重  n个特征 n个一阶权重
        W = tf.get_variable('w1', shape=[numberofFeatures, 1], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.3))
        # 二阶权重 n*n 个   由 Vt*V 构建   V n*k
        V = tf.get_variable('w2', shape=[numberofFeatures, self.K], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.3))
        b = tf.get_variable('b', shape=[1, 1], dtype=tf.float32, initializer=tf.zeros_initializer())

        # 一阶部分
        predict = tf.matmul(X, W) + b

        # 二阶fm
        predict += tf.reduce_sum(tf.square(tf.matmul(X, V)) - tf.matmul(tf.square(X), tf.square(V)),
                                 reduction_indices=1, keep_dims=True) / 2

        #   logistic 损失函数
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=predict))
            tf.summary.scalar("loss", loss)
        # 预测的概率
        y_hat = tf.nn.sigmoid(predict)

        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

        with tf.Session() as sess:
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter('logs/', sess.graph)
            sess.run(tf.global_variables_initializer())
            for i in range(7000):
                    loss_, _, y_h = sess.run([loss, optimizer, y_hat], feed_dict={X: self.x_train, y: self.y_train})

                    if i % 100 == 0:
                        print("train accuracy is %.5f  loss is %.5f" % (
                            accuracy_score(self.y_train.reshape(-1, ),
                                           np.where(np.array(y_h).reshape(-1, ) > 0.5, 1, 0)), loss_))
                    if i % 50 == 0:
                        result = sess.run(merged, feed_dict={X: self.x_train, y: self.y_train})
                        writer.add_summary(result, i)

            y_predict = sess.run(y_hat, feed_dict={X: self.x_test})
            predictions = np.where(np.array(y_predict).reshape(-1, ) > 0.5, 1, 0)
            df = pd.read_csv('./Dataset/test.csv')
            ids = df['PassengerId']
            output = pd.DataFrame({'PassengerId': ids, 'Survived': predictions})
            output.to_csv('result.csv', index=False)


if __name__ == '__main__':
    fm = FM("./Dataset/train.csv", "./Dataset/test.csv", k=3)
    fm.train()
