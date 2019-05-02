import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

MNIST_data_folder = "Distributed Mnist"
mnist = input_data.read_data_sets(MNIST_data_folder, one_hot=True)

worker_num = 10  # 用户数量

batch_size = 100
n_batch = mnist.train.num_examples // batch_size  # //表整除

lr = 0.2
opt = tf.train.GradientDescentOptimizer(lr)


class worker:
    def __init__(self):
        self.w = tf.Variable(tf.zeros([784, 10]))
        self.b = tf.Variable(tf.zeros([10]))
        self.x = tf.placeholder(tf.float32, [None, 784])  # 图片
        self.y = tf.placeholder(tf.float32, [None, 10])  # 标签
        self.prediction = tf.nn.softmax(tf.matmul(self.x, self.w) + self.b)
        self.loss = tf.reduce_mean(tf.square(self.y - self.prediction))
        self.Grad_var_s = opt.compute_gradients(self.loss, var_list=[self.w, self.b])


workers = [worker_num]
for _ in range(worker_num):
    workers_temp = worker()
    workers.append(workers_temp)


class ps:
    def __init__(self):
        self.w = tf.Variable(tf.zeros([784, 10]))
        self.b = tf.Variable(tf.zeros([10]))
        self.x = tf.placeholder(tf.float32, [None, 784])  # 图片
        self.y = tf.placeholder(tf.float32, [None, 10])  # 标签
        self.prediction = tf.nn.softmax(tf.matmul(self.x, self.w) + self.b)
        self.loss = tf.reduce_mean(tf.square(self.y - self.prediction))
        self.avgGradsVars_pre = opt.compute_gradients(self.loss, var_list=[self.w, self.b])
        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.prediction, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.towerGradsVars = []
        for i in range(worker_num):
            self.towerGradsVars.append(workers[i + 1].Grad_var_s)
        self.avgGradsVars = self.average_tower_grads(self.towerGradsVars, self.avgGradsVars_pre)
        self.train_step = opt.apply_gradients(self.avgGradsVars)

    def average_tower_grads(self, tower_grads_vars, avgGradsVars):
        # avgGradsVars
        print('towerGrads:')
        idx = 0
        for grads_vars in tower_grads_vars:  # grads 为 一个list，其中元素为 梯度-变量 组成的二元tuple
            print('grads---tower_%d' % idx)
            for g_var in grads_vars:
                print(g_var)
            idx += 1
        for i in range(len(avgGradsVars)):
            avgGradsVars[i] = list(avgGradsVars[i])
            avgGradsVars[i][0] = tf.reduce_mean([tower_grads_vars[j][i][0] for j in range(idx)], 0)
            avgGradsVars[i] = tuple(avgGradsVars[i])
        # print('test', avgGradsVars)
        return avgGradsVars




ps = ps()

# 初始化变量
init = tf.global_variables_initializer()

update_W = [worker_num]
update_b = [worker_num]
for i in range(worker_num):
    update_W.append(tf.assign(workers[i + 1].w, ps.w))
    update_b.append(tf.assign(workers[i + 1].b, ps.b))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(31):
        for batch in range(n_batch):
            batch_xs = [worker_num]  # 初始化
            batch_ys = [worker_num]  # 初始化
            for i in range(worker_num):
                x_temp, y_temp = mnist.train.next_batch(batch_size // worker_num)
                batch_xs.append(x_temp)
                batch_ys.append(y_temp)
            feed_dict = {}
            for i in range(worker_num):
                feed_dict[workers[i + 1].x] = batch_xs[i + 1]
                feed_dict[workers[i + 1].y] = batch_ys[i + 1]
            sess.run(ps.train_step, feed_dict=feed_dict)
            # print('1w', sess.run(workers[1].Grad_var_s, feed_dict=feed_dict))
            # print('2w', sess.run(workers[2].Grad_var_s, feed_dict=feed_dict))
            # print('psw', sess.run(ps.avgGradsVars, feed_dict=feed_dict))
            for i in range(worker_num):
                sess.run([update_W[i + 1], update_b[i + 1]])
        acc = sess.run(ps.accuracy, feed_dict={ps.x: mnist.test.images, ps.y: mnist.test.labels})
        print("Round " + str(epoch) + ", Testing Accuracy " + str(acc))
