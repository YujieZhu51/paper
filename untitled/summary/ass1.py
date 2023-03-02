import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 批次（具体含义？）
batch_size = 100
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 28 * 28])
y = tf.placeholder(tf.float32, [None, 10])

# weight和bias
w1 = tf.Variable(tf.truncated_normal([28 * 28, 400], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape= [400]))
# 输出
h = tf.nn.relu(tf.matmul(x, w1) + b1)

# 第二层
w2 = tf.Variable(tf.truncated_normal([400, 10], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[10]))

# activation function
results = tf.nn.softmax(tf.matmul(h, w2) + b2)

# loss函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=results))

# 梯度下降(0.2这个数？怎么调)
train = tf.train.GradientDescentOptimizer(learning_rate=0.2).minimize(loss)

init = tf.global_variables_initializer()

correct = tf.equal(tf.argmax(y, 1), tf.argmax(results, 1))

accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(25):
        for batch in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: batch_x, y: batch_y})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("迭代" + str(epoch) + " acc: " + str(acc))
