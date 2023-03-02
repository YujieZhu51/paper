import tensorflow as tf
import os
import random

# ----PNEUMONIA-----------------------------------
train_nor_path="D://deep_learning//PNEUMONIA_DATA//train//NORMAL"
train_p_path="D://deep_learning//PNEUMONIA_DATA//train//PNEUMONIA"
train_nor_file_name=[]
train_p_file_name=[]
train_image_path=[]
train_label=[]
for filename in os.listdir(train_nor_path):
    train_nor_file_name.append(filename)
for filename in os.listdir(train_p_path):
    train_p_file_name.append(filename)
for filename in train_p_file_name:
    train_image_path.append(os.path.join(train_p_path, filename))
    if "bacteria" in filename:
        train_label.append([0,1,0])
    if "virus" in filename:
        train_label.append([0,0,1])
for filename in train_nor_file_name:
    train_image_path.append(os.path.join(train_nor_path, filename))
    train_label.append([1,0,0])

train_label = tf.convert_to_tensor(train_label, dtype=tf.float32)

trainImage = []

for image_path in train_image_path:
    image_raw_data = tf.gfile.FastGFile(image_path, 'rb').read() #‘r’：UTF-8编码； ‘rb’：非UTF-8编码
    image_data = tf.image.decode_jpeg(image_raw_data,channels=1)
    image_data = tf.image.resize_images(image_data, [96,128], 0)
    # 一共有三个参数，第一个是原始图像，第二个是重采样目标影像大小，第三个是重采样的方法。双线性插值算法（Bilinear interpolation）;Method取值为：0；最近邻居法（Nearest  neighbor interpolation);Method取值为：1；双三次插值法（Bicubic interpolation);Method取值为：2
    trainImage.append(image_data)
trainImage = tf.cast(trainImage,dtype=tf.float32)
#trainImage = tf.divide(trainImage,255)


# -----分类---------------------------------
def next_batch(batch_size):
    x1=[]
    y1=[]
    rs = random.sample(range(0, 500), batch_size)
    for r in rs:
        x1.append(trainImage[r])
        y1.append(train_label[r])
    x1 = tf.convert_to_tensor(x1, dtype=tf.float32)
    y1 = tf.convert_to_tensor(y1, dtype=tf.float32)
    x1=x1.eval()
    y1=y1.eval()

    return x1,y1


batch_size = 50
n_batch = 500 // batch_size

x = tf.placeholder(tf.float32, [None, 96, 128, 1])
y = tf.placeholder(tf.float32, [None, 3])

w1 = tf.Variable(tf.truncated_normal([3,3,1,16], stddev=0.1),name='w1')
b1 = tf.Variable(tf.constant(0.1, shape= [16]),name='b1')
c1 = tf.nn.conv2d(x,w1,strides=[1,1,1,1],padding='SAME') + b1
h_conv1 = tf.nn.relu(c1)

h_pool1 = tf.nn.max_pool(h_conv1,ksize = [1,4,4,1],strides=[1,4,4,1],padding='SAME')

w2 = tf.Variable(tf.truncated_normal([3,3,16,32], stddev=0.1),name='w1')
b2 = tf.Variable(tf.constant(0.1, shape= [32]),name='b1')
c2 = tf.nn.conv2d(h_pool1,w2,strides=[1,1,1,1],padding='SAME') + b2
h_conv2 = tf.nn.relu(c2)

h_pool2 = tf.nn.max_pool(h_conv2,ksize = [1,4,4,1],strides=[1,4,4,1],padding='SAME')

h_flat = tf.reshape(h_pool2,[-1,6 * 8 * 32])
w_full1 = tf.Variable(tf.truncated_normal([6 * 8 * 32,64], stddev=0.1),name='w_full1')
b_full1 = tf.Variable(tf.constant(0.1, shape= [64]),name='b_full1')
h_full1 = tf.nn.relu(tf.matmul(h_flat, w_full1) + b_full1)

keep_pro = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_full1,keep_pro)

w_full2 = tf.Variable(tf.truncated_normal([64, 3], stddev=0.1),name='w_full2') # 人就改成5
b_full2 = tf.Variable(tf.constant(0.1, shape=[3]),name='b_full2')
y_pred = tf.nn.softmax(tf.matmul(h_fc1_drop, w_full2) + b_full2)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(7):
        count=0
        for batch in range(n_batch):
            count = count+1
            batch_x, batch_y = next_batch(batch_size)
            sess.run(train, feed_dict={x: batch_x, y: batch_y, keep_pro: 0.5})
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_pro: 0.5})
            print(str(count) + " acc: " + str(acc))
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_pro: 0.5})
        print("迭代" + str(epoch+1) + " acc: " + str(acc))





