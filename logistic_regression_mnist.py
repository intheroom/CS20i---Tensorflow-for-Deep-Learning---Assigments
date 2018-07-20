import tensorflow as tf
import utils
from tensorflow.examples.tutorials.mnist import input_data

NUM_EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001

mnist_folder = 'data/mnist'
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)

# load MNIST data
mnist_train = tf.data.Dataset.from_tensor_slices(train).shuffle(5000).batch(BATCH_SIZE)

mnist_test = tf.data.Dataset.from_tensor_slices(test).batch(BATCH_SIZE)

iterator = tf.data.Iterator.from_structure(mnist_train.output_types, mnist_train.output_shapes)
feature, label = iterator.get_next()

train_init = iterator.make_initializer(mnist_train)
test_init = iterator.make_initializer(mnist_test)

training_size = 55000
test_size = 10000
training_batch_nums = training_size // BATCH_SIZE + 1
test_batch_nums = test_size // BATCH_SIZE + 1

W1 = tf.get_variable(shape=(784,100), initializer=tf.random_normal_initializer(0,0.01), name='W1')
b1 = tf.get_variable(shape=(1,100), initializer=tf.zeros_initializer(), name='b1')
W2 = tf.get_variable(shape=(100,10), initializer=tf.random_normal_initializer(0,0.01), name='W2')
b2 = tf.get_variable(shape=(1, 10), initializer=tf.zeros_initializer(), name='b2')

Z1 = tf.add(tf.matmul(feature, W1), b1, name='affine_1')
A1 = tf.nn.relu(Z1, name='relu_1')
Z2 = tf.add(tf.matmul(A1, W2), b2, name='affine_2')

out = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z2, labels=label, name='out')
loss = tf.reduce_mean(out, name='loss')

optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

preds = tf.nn.softmax(Z2)
correct_preds = tf.equal(tf.argmax(preds, axis=1), tf.argmax(label, axis=1))
correct_nums = tf.reduce_sum(tf.cast(correct_preds, tf.int32))

writer = tf.summary.FileWriter('./graphs/logistic_regression', tf.get_default_graph())

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(NUM_EPOCHS):
		sess.run(train_init)
		total_loss = 0
		for j in range(training_batch_nums):
			batch_loss,_ = sess.run([loss, optimizer])
			total_loss += batch_loss
		print('Average epoch loss: ', i, total_loss / training_batch_nums)
	print('Training complete')

	sess.run(test_init)
	total_correct_nums = 0
	for i in range(test_batch_nums):
		total_correct_nums += sess.run(correct_nums)
		# print(sess.run([correct_nums]))
	accuracy = total_correct_nums * 1.0 / test_size
	print('Accuracy: ', accuracy)
writer.close()
