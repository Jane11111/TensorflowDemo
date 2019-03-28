# -*- coding: utf-8 -*-
# @Time    : 2019-03-22 15:25
# @Author  : zxl
# @FileName: SimpleNN.py


import numpy as np
import pandas as pd
import tensorflow as tf
import urllib.request as request
import matplotlib.pyplot as plt

IRIS_TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

names=['sepal-length','sepl-width','petal-length','petal-width','species']
train=pd.read_csv(IRIS_TRAIN_URL,names=names,skiprows=1)
test=pd.read_csv(IRIS_TEST_URL,names=names,skiprows=1)

Xtrain=train.drop("species",axis=1)
Xtest=test.drop("species",axis=1)

ytrain=pd.get_dummies(train.species)
ytest=pd.get_dummies(test.species)

def create_train_model(hidden_nodes,num_iters):

    tf.reset_default_graph()

    #输入和输出
    X=tf.placeholder(shape=(120,4),dtype=tf.float64,name='X')
    y=tf.placeholder(shape=(120,3),dtype=tf.float64,name='y')

    #权重矩阵
    W1=tf.Variable(np.random.rand(4,hidden_nodes),dtype=tf.float64)
    W2=tf.Variable(np.random.rand(hidden_nodes,3),dtype=tf.float64)

    #
    A1=tf.sigmoid(tf.matmul(X,W1))
    y_est=tf.sigmoid(tf.matmul(A1,W2))

    #损失函数
    deltas=tf.square(y_est-y)
    loss=tf.reduce_sum(deltas)#目的是为了降低损失函数

    #梯度下降方法降低损失函数
    optimizer=tf.train.GradientDescentOptimizer(0.005)
    train=optimizer.minimize(loss)

    #初始化变量
    init=tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)

    #多次迭代
    for i in range(num_iters):
        sess.run(train,feed_dict={X:Xtrain,y:ytrain})
        loss_plot[hidden_nodes].append(sess.run(loss,feed_dict={
            X:Xtrain.as_matrix(),y:ytrain.as_matrix()
        }))
        weights1=sess.run(W1)#训练这个矩阵
        weights2=sess.run(W2)#训练第2个矩阵

        """
        如果我每次迭代分成很多份呢，每个人的权重矩阵为一个单位
        针对于每个人，只训练它的连接矩阵？
        最后想要的也就是这么多矩阵而已呀
        
        """

    print("loss(hidden nodes: %d,iterations: %d): %.2f"%(hidden_nodes,num_iters,loss_plot[hidden_nodes][-1]))
    sess.close()
    return weights1,weights2

num_hidden_nodes=[5,10,20]
loss_plot={5:[],10:[],20:[]}
weights1={5:None,10:None,20:None}
weights2={5:None,10:None,20:None}
num_iters=10

#可视化损失函数
# plt.figure(figsize=(12,8))
for hidden_nodes in num_hidden_nodes:
    weights1[hidden_nodes],weights2[hidden_nodes]=create_train_model(hidden_nodes,num_iters)
#     plt.plot(range(num_iters),loss_plot[hidden_nodes],label="nn:4-%d-3"%(hidden_nodes))
#
# plt.xlabel("Iteration",fontsize=12)
# plt.ylabel("Loss",fontsize=12)
# plt.legend(fontsize=12)
# plt.show()

#Evaluate
X=tf.placeholder(shape=(30,4),dtype=tf.float64,name='X')
# y=tf.placeholder(shape=(30,3),dtype=tf.float64,name='y')

for hidden_nodes in num_hidden_nodes:

    #前向传播
    W1=tf.Variable(weights1[hidden_nodes])
    W2=tf.Variable(weights2[hidden_nodes])
    A1=tf.sigmoid(tf.matmul(X,W1))
    y_est=tf.sigmoid(tf.matmul(A1,W2))

    #计算预测输出
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        y_est_np=sess.run(y_est,feed_dict={X:Xtest})
        a=3

    correct=[estimate.argmax(axis=0)==target.argmax(axis=0)
             for estimate, target in zip(y_est_np,ytest.as_matrix())]

    accuracy=100*sum(correct)/len(correct)
    print('Network architecture 4-%d-3, accuracy: %.2f'%(hidden_nodes,accuracy))





