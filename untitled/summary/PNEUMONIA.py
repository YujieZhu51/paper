import tensorflow as tf
import os
import random
import numpy as np
import cv2

# ----PNEUMONIA-----------------------------------
#maxx=500
train_nor_path="C://deep_learning//PNEUMONIA DATA//PNEUMONIA DATA//train//NORMAL"
train_p_path="C://deep_learning//PNEUMONIA DATA//PNEUMONIA DATA//train//PNEUMONIA"
test_nor_path="C://deep_learning//PNEUMONIA DATA//PNEUMONIA DATA//test//NORMAL"
test_p_path="C://deep_learning//PNEUMONIA DATA//PNEUMONIA DATA//test//PNEUMONIA"
train_nor_file_name=[]
train_p_file_name=[]
train_image_path=[]
train_label=[]
test_nor_file_name=[]
test_p_file_name=[]
test_image_path=[]
test_label=[]
for filename in os.listdir(train_nor_path):
    train_nor_file_name.append(filename)
for filename in os.listdir(train_p_path):
    train_p_file_name.append(filename)


for filename in os.listdir(test_nor_path):
    test_nor_file_name.append(filename)
for filename in os.listdir(test_p_path):
    test_p_file_name.append(filename)

for filename in train_p_file_name:
    train_image_path.append(os.path.join(train_p_path, filename))
    if "bacteria" in filename:
        train_label.append([0,1,0])
    if "virus" in filename:
        train_label.append([0,0,1])
for filename in train_nor_file_name:
    train_image_path.append(os.path.join(train_nor_path, filename))
    train_label.append([1,0,0])

for filename in test_p_file_name:
    test_image_path.append(os.path.join(test_p_path, filename))
    if "bacteria" in filename:
        test_label.append([0,1,0])
    if "virus" in filename:
        test_label.append([0,0,1])
for filename in test_nor_file_name:
    test_image_path.append(os.path.join(test_nor_path, filename))
    test_label.append([1,0,0])

train_label=np.array(train_label)
test_label=np.array(test_label)
#train_label = tf.convert_to_tensor(train_label, dtype=tf.float32)
#test_label = tf.convert_to_tensor(test_label, dtype=tf.float32)

trainImage = []
testImage = []


for image_path in train_image_path:
    img = cv2.imread(image_path, 0)
    res = cv2.resize(img, dsize=(64, 48), interpolation=cv2.INTER_LINEAR)
    res = res.reshape(48, 64, 1)
    trainImage.append(res)
#trainImage = tf.cast(trainImage,dtype=tf.float32)
#trainImage = tf.divide(trainImage,255)
trainImage=np.array(trainImage)

for image_path in test_image_path:
    img = cv2.imread(image_path, 0)
    res = cv2.resize(img, dsize=(64, 48), interpolation=cv2.INTER_LINEAR)
    res = res.reshape(48, 64, 1)
    testImage.append(res)
testImage=np.array(testImage)

print('image down')
# -----分类---------------------------------
def next_batch(batch_size):
    x1=[]
    y1=[]
    rs = random.sample(range(0, 5231), batch_size)
    for r in rs:
        x1.append(trainImage[r])
        y1.append(train_label[r])
    #x1 = tf.convert_to_tensor(x1, dtype=tf.float32)
    #y1 = tf.convert_to_tensor(y1, dtype=tf.float32)
    #x1=x1.eval()
    #y1=y1.eval()

    return x1,y1


batch_size = 100
n_batch = 5232 // batch_size

x = tf.placeholder(tf.float32, [None, 48, 64, 1])
y = tf.placeholder(tf.float32, [None, 3])

ww1 = tf.Variable(tf.truncated_normal([3,3,1,16], stddev=0.1),name='ww1')
bb1 = tf.Variable(tf.constant(0.1, shape= [16]),name='bb1')
c1 = tf.nn.conv2d(x,ww1,strides=[1,1,1,1],padding='SAME') + bb1
wx1 = tf.Variable(tf.truncated_normal([1,1,1,16], stddev=0.1),name='wx1')
bx1 = tf.Variable(tf.constant(0.1, shape= [16]),name='bx1')
x1 = tf.nn.conv2d(x,wx1,strides=[1,1,1,1],padding='SAME') + bx1
add1 = tf.add(c1, x1)
nor1=tf.layers.batch_normalization(add1,training=True)
h_conv1 = tf.nn.relu(nor1)

ww12 = tf.Variable(tf.truncated_normal([3,3,16,32], stddev=0.1),name='ww12')
bb12 = tf.Variable(tf.constant(0.1, shape= [32]),name='bb12')
c12 = tf.nn.conv2d(h_conv1,ww12,strides=[1,1,1,1],padding='SAME') + bb12
wx12 = tf.Variable(tf.truncated_normal([1,1,16,32], stddev=0.1),name='wx12')
bx12 = tf.Variable(tf.constant(0.1, shape= [32]),name='bx12')
x12 = tf.nn.conv2d(h_conv1,wx12,strides=[1,1,1,1],padding='SAME') + bx12
add12 = tf.add(c12, x12)
nor12=tf.layers.batch_normalization(add12,training=True)
h_conv12 = tf.nn.relu(nor12)


h_pool1 = tf.nn.max_pool(h_conv12,ksize = [1,4,4,1],strides=[1,4,4,1],padding='SAME')

ww2 = tf.Variable(tf.truncated_normal([3,3,32,64], stddev=0.1),name='ww2')
bb2 = tf.Variable(tf.constant(0.1, shape= [64]),name='bb2')
c2 = tf.nn.conv2d(h_pool1,ww2,strides=[1,1,1,1],padding='SAME') + bb2
wx2 = tf.Variable(tf.truncated_normal([1,1,32,64], stddev=0.1),name='wx2')
bx2 = tf.Variable(tf.constant(0.1, shape= [64]),name='bx2')
x2 = tf.nn.conv2d(h_pool1,wx2,strides=[1,1,1,1],padding='SAME') + bx2
add2 = tf.add(c2, x2)
nor2=tf.layers.batch_normalization(add2,training=True)
h_conv2 = tf.nn.relu(nor2)

ww22 = tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.1),name='ww22')
bb22 = tf.Variable(tf.constant(0.1, shape= [128]),name='bb22')
c22 = tf.nn.conv2d(h_conv2,ww22,strides=[1,1,1,1],padding='SAME') + bb22
wx22 = tf.Variable(tf.truncated_normal([1,1,64,128], stddev=0.1),name='wx22')
bx22 = tf.Variable(tf.constant(0.1, shape= [128]),name='bx22')
x22 = tf.nn.conv2d(h_conv2,wx22,strides=[1,1,1,1],padding='SAME') + bx22
add22 = tf.add(c22, x22)
nor22=tf.layers.batch_normalization(add22,training=True)
h_conv22 = tf.nn.relu(nor22)

h_pool2 = tf.nn.max_pool(h_conv22,ksize = [1,4,4,1],strides=[1,4,4,1],padding='SAME')

h_flat = tf.reshape(h_pool2,[-1,3 * 4 * 128])
w_full1 = tf.Variable(tf.truncated_normal([ 3 * 4 * 128,128], stddev=0.1),name='w_full1')
b_full1 = tf.Variable(tf.constant(0.1, shape= [128]),name='b_full1')
h_full1 = tf.nn.relu(tf.matmul(h_flat, w_full1) + b_full1)

keep_pro = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_full1,keep_pro)

w_full2 = tf.Variable(tf.truncated_normal([128, 3], stddev=0.1),name='w_full2') # 人就改成5
b_full2 = tf.Variable(tf.constant(0.1, shape=[3]),name='b_full2')
y_pred = tf.nn.softmax(tf.matmul(h_fc1_drop, w_full2) + b_full2)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
train = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(25):
        count=0
        for batch in range(n_batch):
            count = count+1
            batch_x, batch_y = next_batch(batch_size)
            sess.run(train, feed_dict={x: batch_x, y: batch_y, keep_pro: 0.5})
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_pro: 0.5})
            print(str(count) + " acc: " + str(acc))
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_pro: 0.5})
        print("迭代" + str(epoch+1) + " acc: " + str(acc))
    acc2 = sess.run(accuracy, feed_dict={x: testImage, y: test_label, keep_pro: 1})
    print("test" + " acc: " + str(acc2))
    #saver = tf.train.Saver()
    #saver.save(sess, ".//PNEUMONIA")




