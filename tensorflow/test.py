import tensorflow as tf
import numpy as np

#names are for saving and restoring tf variables

#tensorboard
x = tf.constant(3.0, 'float32')
y = tf.constant(4, 'float32')

x_var = tf.Variable(3.1, name="x_variable")
y_var = tf.Variable(2.3, name="y_variable")

#4x4 matrix
matrix1_var = tf.Variable(tf.truncated_normal([4,4], mean=0.0, stddev=1.0))
matrix2_var = tf.Variable(tf.truncated_normal([4,4], mean=1.0, stddev=0.5))

#added
matrix3 = matrix1_var + matrix2_var
multiplied = matrix1_var * matrix2_var

#init
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

print(sess.run(matrix3))
print(sess.run(multiplied))

file_writer = tf.summary.FileWriter('logs/', sess.graph)
sess.close()
