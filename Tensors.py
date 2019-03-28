
import tensorflow as tf

tf.reset_default_graph()

a=tf.placeholder("float",name="pholdA")
print("a:",a)

b=tf.Variable(2.0,name="varB")
print("b:",b)

c=tf.constant([1.,2.,3.,4.],name="consC")
print("c:",c)

d=a*b+c
print(d)

graph=tf.get_default_graph()
for op in graph.get_operations():
    print(op.name)

init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)
print(sess.run(d,feed_dict={a:[[0.5],[2],[3]]}))