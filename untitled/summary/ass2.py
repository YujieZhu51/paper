import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batch_size = 100
n_batch = mnist.train.num_examples // batch_size
x = tf.placeholder(tf.float32, [None, 28 * 28])
y = tf.placeholder(tf.float32, [None, 10])
x_4d = tf.reshape(x,[-1,28,28,1])
'''
# 卷积核
w1 = tf.Variable(tf.truncated_normal([3,3,1,16], stddev=0.1),name='w1') # 窗口是3*3的，通道数1，16个卷积核
b1 = tf.Variable(tf.constant(0.1, shape= [16]),name='b1')

# 把x转化成四维的
x_4d = tf.reshape(x,[-1,28,28,1]) # -1是它自己能算出来是多少（图片的数量），最后一个1是每一个像素位点的通道为1

# 卷积操作
c1 = tf.nn.conv2d(x_4d,w1,strides=[1,1,1,1],padding='SAME') + b1 # strides步长[1,x,y,1] padding是方式，same是外面补零的方法
# 激活函数
h_conv1 = tf.nn.relu(c1)

# 池化
h_pool1 = tf.nn.max_pool(h_conv1,ksize = [1,2,2,1],strides=[1,2,2,1],padding='SAME') # kernal size是多大的区域选一个最大值

#第二层卷积
w2 = tf.Variable(tf.truncated_normal([3,3,16,32], stddev=0.1),name='w2') # 进来16出去32
b2 = tf.Variable(tf.constant(0.1, shape= [32]),name='b2')

c2 = tf.nn.conv2d(h_pool1,w2,strides=[1,1,1,1],padding='SAME') + b2

h_conv2 = tf.nn.relu(c2)

h_pool2 = tf.nn.max_pool(h_conv2,ksize = [1,2,2,1],strides=[1,2,2,1],padding='SAME')
# -----------------------全连接层--------------------------------------------------------

# 把输出变成一维
h_flat = tf.reshape(h_pool2,[-1,7 * 7 * 32])

# 初始w和b
w_full1 = tf.Variable(tf.truncated_normal([7 * 7 * 32, 512], stddev=0.1),name='w_full1')
b_full1 = tf.Variable(tf.constant(0.1, shape= [512]),name='b_full1')

#一层输出
h_full1 = tf.nn.relu(tf.matmul(h_flat, w_full1) + b_full1)

# 第二层
w_full2 = tf.Variable(tf.truncated_normal([512, 10], stddev=0.1),name='w_full2')
b_full2 = tf.Variable(tf.constant(0.1, shape=[10]),name='b_full2')

# activation function
y_pred = tf.nn.softmax(tf.matmul(h_full1, w_full2) + b_full2)

# loss函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))

# 梯度下降
train = tf.train.GradientDescentOptimizer(learning_rate=0.2).minimize(loss)
'''
w1 = tf.Variable(tf.truncated_normal([3,3,1,16], stddev=0.1),name='w1')
b1 = tf.Variable(tf.constant(0.1, shape= [16]),name='b1')
c1 = tf.nn.conv2d(x_4d,w1,strides=[1,1,1,1],padding='SAME') + b1
h_conv1 = tf.nn.relu(c1)
# ----------CNN第二层-----------------------------

w2 = tf.Variable(tf.truncated_normal([3,3,16,32], stddev=0.1),name='w2')
b2 = tf.Variable(tf.constant(0.1, shape= [32]),name='b2')
c2 = tf.nn.conv2d(h_conv1,w2,strides=[1,1,1,1],padding='SAME') + b2
h_conv2 = tf.nn.relu(c2)

# ----------池化-----------------------------
h_pool1 = tf.nn.max_pool(h_conv2,ksize = [1,2,2,1],strides=[1,2,2,1],padding='SAME')


# ----------CNN第三层-----------------------------
w3 = tf.Variable(tf.truncated_normal([3,3,32,64], stddev=0.1),name='w3')
b3 = tf.Variable(tf.constant(0.1, shape= [64]),name='b3')
c3 = tf.nn.conv2d(h_pool1,w3,strides=[1,1,1,1],padding='SAME') + b3
h_conv3 = tf.nn.relu(c3)
# ----------CNN第四层-----------------------------

w4 = tf.Variable(tf.truncated_normal([3,3,64,64], stddev=0.1),name='w4')
b4 = tf.Variable(tf.constant(0.1, shape= [64]),name='b4')
c4 = tf.nn.conv2d(h_conv3,w4,strides=[1,1,1,1],padding='SAME') + b4
h_conv4 = tf.nn.relu(c4)

# ----------池化-----------------------------
h_pool2 = tf.nn.max_pool(h_conv4,ksize = [1,2,2,1],strides=[1,2,2,1],padding='SAME')


# ----------全连接-----------------------------
h_flat = tf.reshape(h_pool2,[-1,7 * 7 * 64])
w_full1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 256], stddev=0.1),name='w_full1')
b_full1 = tf.Variable(tf.constant(0.1, shape= [256]),name='b_full1')
h_full1 = tf.nn.relu(tf.matmul(h_flat, w_full1) + b_full1)

keep_pro = 0.7
h_fc1_drop = tf.nn.dropout(h_full1,keep_pro)

w_full2 = tf.Variable(tf.truncated_normal([256, 10], stddev=0.1),name='w_full2') # 人就改成5
b_full2 = tf.Variable(tf.constant(0.1, shape=[10]),name='b_full2')
init = tf.global_variables_initializer()
y_pred = tf.nn.softmax(tf.matmul(h_fc1_drop, w_full2) + b_full2)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
train =tf.train.GradientDescentOptimizer(learning_rate=0.2).minimize(loss)
#准确率
correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))

accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(3):
        for batch in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: batch_x, y: batch_y})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("迭代" + str(epoch+1) + " acc: " + str(acc))
    saver = tf.train.Saver()
    saver.save(sess, './/ass4')  # global_step可以改名字