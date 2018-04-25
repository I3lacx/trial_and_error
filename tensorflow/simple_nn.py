from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

print(mnist)

#Parameters
learning_rate = 0.1
num_step = 5000
batch_size = 128
display_step = 100

input_layer = 256
n_hidden_1 = 192
n_hidden_2 = 98
num_input = 784
num_classes = 10

X = tf.placeholder('float32', [None, num_input])
Y = tf.placeholder( 'float32', [None, num_classes])

weights = {
	'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}

biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'b2': tf.Variable(tf.random_normal([n_hidden_2]))
}

#model:

def neural_net(x):
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	#activation function?
	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])

	layer_out = tf.matmul(layer_2, weights['out'])
	return layer_out

logits = neural_net(X)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#switch it to maxmize
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for step in range(1,num_step):
		batch_x, batch_y = mnist.train.next_batch(batch_size)

		sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
		if step % display_step == 0 or step == 1:
			loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
			                                                     Y: batch_y})
			print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

	print("finished")

	print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))
