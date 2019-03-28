# -*- coding: utf-8 -*-
# @Time    : 2019-03-22 17:22
# @Author  : zxl
# @FileName: AMLPWithInter.py

import tensorflow as tf
import numpy as np

#TODO 需要更改
stu_num=1
n1=60
n2=8
num_iters=100
train_X1,test_X1=[],[]
train_X2,test_X2=[],[]
train_y,test_y=[],[]
class AMLPWithInter:


    """
    h:隐层单元个数
    num_iters:迭代次数
    """
    def __init__(self,h,num_iters):
        self.h=h
        self.num_iters=num_iters

    def fit(self,X1,X2,Y):
        (Weights1,Weights_u,Weights2)=self.create_train_model(X1,X2,Y)
        self.Weights1=Weights1
        self.Weights_u=Weights_u
        self.Weights2=Weights2


    def predict(self,test_X1,test_X2):
        stu_num=len(test_X1)
        time_span=len(test_X1[0])
        res=list()
        #为每一个同学建一个图？
        #TODO 是否有更好的解决办法？
        for u in range(stu_num):
            # 输入输出数据
            X1 = tf.placeholder(shape=(time_span, n1), dtype=tf.float64, name='X1')
            X2 = tf.placeholder(shape=(time_span, n2), dtype=tf.float64, name='X2')

            W1=tf.Variable(self.Weights1)
            Wu=tf.Variable(self.Weights_u[u])
            W2=tf.Variable(self.Weights2)
            #TODO 激活函数是否需要修改
            A1=tf.sigmoid(tf.matmul(X1,W1))
            Au=tf.sigmoid(tf.matmul(X2,Wu))

            #交叉层
            A2=tf.multiply(A1,Au)

            #预测值
            y_est=tf.matmul(A2,W2)


            init=tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                cur_y_est=sess.run(y_est,feed_dict={X1:test_X1[u],X2:test_X2[u],})
                res.append(cur_y_est.flatten())

        return res







    def create_train_model(self,train_X1,train_X2,train_y):


        h1=self.h
        h2=self.h
        time_span=len(train_X1[0])

        """
        每个人当成一个单位来训练吧
        """

        tf.reset_default_graph()

        #输入输出数据
        X1=tf.placeholder(shape=(time_span,n1),dtype=tf.float64,name='X1')
        X2=tf.placeholder(shape=(time_span,stu_num*n2),dtype=tf.float64,name='X2')
        Y=tf.placeholder(shape=(time_span,1),dtype=tf.float64,name='Y')

        #权重矩阵
        #TODO 这里的处理可以参照预测时候，每个人的公有权重矩阵在训练时候进行传递
        #感觉可以。
        W1=tf.Variable(np.random.rand(n1,h1))
        Wu=list()
        for u in range(stu_num):
            cur_Wu=tf.Variable(np.random.rand(n2,h2))
            Wu.append(cur_Wu)

        W2=tf.Variable(np.random.rand(h1,1))

        #创建图结构
        #TODO 公有结构隐层激活函数
        A1=tf.sigmoid(tf.matmul(X1,W1))
        Au_list=list()
        for u in range(stu_num):
            #TODO 似有结构隐层激活函数
            cur_Au=tf.sigmoid(tf.matmul(X2[u*time_span:(u+1)*time_span],Wu[u]))
            Au_list.append(cur_Au)
        Au=Au_list[0]
        for i in np.arange(1,stu_num,1):
            Au+=Au_list[i]

        #交叉层,需要确保h1=h2
        A2=tf.multiply(A1,Au)

        #TODO 输出层激活函数
        y_est=tf.sigmoid(tf.matmul(A2,W2))

        #损失函数
        deltas=tf.square(y_est-Y)
        loss=tf.reduce_sum(deltas)

        #最小化损失函数
        #TODO 步长需要改
        optimizer=tf.train.GradientDescentOptimizer(0.005)
        train=optimizer.minimize(loss)

        #初始化参数
        init=tf.global_variables_initializer()
        sess=tf.Session()
        sess.run(init)

        Weights1=None
        Weights_u=list()
        for u in range(stu_num):
            Weights_u.append(np.full(shape=(n2,h2),fill_value=0))
        Weights2=None

        for i in range(self.num_iters):
            #以每一个学生为一个单位进行训练
            for u in range(stu_num):
                Xu1=train_X1[u]
                Xu2=train_X2[u]
                y=train_y[u]
                sess.run(train,feed_dict={X1:Xu1,X2:Xu2,Y:y})
                Weights1=sess.run(W1)
                Weights_u[u]=sess.run(Wu[u])
                Weights2=sess.run(W2)
        sess.close()
        return (Weights1,Weights_u,Weights2)














