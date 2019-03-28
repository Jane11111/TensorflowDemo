import tensorflow as tf

# hello=tf.constant('Hello  Tensorflow!')
# sess=tf.Session()
# print(sess.run(hello))
#
# a=tf.constant(10)
# b=tf.constant(32)
# print(sess.run(a+b))

# sess=tf.Session()
# vec=tf.random_uniform(shape=(3,))
# out1=vec+1
# out2=vec+2
# print(sess.run(vec))
# print(sess.run(vec))
# print(sess.run((out1,out2)))

# x=tf.placeholder(tf.float32)
# y=tf.placeholder(tf.float32)
# z=x+y
# print(sess.run(z,feed_dict={x:3,y:4.5}))
# print(sess.run(z,feed_dict={x:[1,3],y:[2,4]}))

# my_data=[
#     [0,1,],
#     [2,3,],
#     [4,5,],
#     [6,7,],
# ]
# slices=tf.data.Dataset.from_tensor_slices(my_data)
# next_item=slices.make_one_shot_iterator().get_next()
# while True:
#     try:
#         print(sess.run(next_item))
#     except tf.errors.OutOfRangeError:
#         break

# x=tf.placeholder(tf.float32,shape=[None,3])
# # linear_model=tf.layers.Dense(units=1)
# # y=linear_model(x)
# y=tf.layers.dense(x,units=1)
# init=tf.global_variables_initializer()
# sess.run(init)
# print(sess.run(y,{x:[[1,2,3],[4,5,6]]}))

# features={
#     "sales":[[5],[10],[8],[9]],
#     "department":["sports","sports","gardening","gardening"]
# }
#
# department_column=tf.feature_column.categorical_column_with_vocabulary_list(
#     "department",["sports","gardening"]
# )
# department_column=tf.feature_column.indicator_column(
#     department_column
# )
# columns=[
#     tf.feature_column.numeric_column("sales"),
#     department_column
# ]
# inputs=tf.feature_column.input_layer(features,columns)
# var_init=tf.global_variables_initializer()
# table_init=tf.tables_initializer()
# sess.run(var_init,table_init)
# print(sess.run(inputs))

#Define the data
x=tf.constant([[1],[2],[3],[4]],dtype=tf.float32)
y_true=tf.constant([[0],[-1],[-2],[-4]],dtype=tf.float32)
#Define the odel
linear_model=tf.layers.Dense(units=1)
y_pred=linear_model(x)

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
print(sess.run(y_pred))
#Loss
loss=tf.losses.mean_squared_error(labels=y_true,predictions=y_pred)
print(sess.run(loss))
#Training
optimizer=tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss)
for i in range(100):
    _,loss_value=sess.run((train,loss))
    print(loss_value)

print(sess.run(y_pred))

