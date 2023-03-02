import tensorflow as tf
import os
import random
import numpy as np
import cv2

# ----PNEUMONIA-----------------------------------
p0='E://deep_learning//COVID DATA//cut//0//'
p1='E://deep_learning//COVID DATA//cut//1//'
p2='E://deep_learning//COVID DATA//cut//2//'
p3='E://deep_learning//COVID DATA//cut//3//'
filename0=[]
filename1=[]
filename2=[]
filename3=[]
image_path_left=[]
image_path_right=[]
for filename in os.listdir(p0):
    filename0.append(filename)
for filename in os.listdir(p1):
    filename1.append(filename)
for filename in os.listdir(p2):
    filename2.append(filename)
for filename in os.listdir(p3):
    filename3.append(filename)

for filename in filename0:
    image_path_left.append(os.path.join(p0, filename))
for filename in filename1:
    image_path_right.append(os.path.join(p1, filename))
for filename in filename2:
    image_path_left.append(os.path.join(p2, filename))
for filename in filename3:
    image_path_right.append(os.path.join(p3, filename))


Image = []
label = []

for image_path in  image_path_left:
    img = cv2.imread(image_path, 0)
    res = cv2.resize(img, dsize=(32, 48), interpolation=cv2.INTER_LINEAR)
    res = res.reshape(48, 32, 1)
    a1 = res[::-1]
    Image.append(res)
    Image.append(a1)
    label.append([1, 0, 0, 0])
    label.append([0, 1, 0, 0])

for image_path in  image_path_right:
    img = cv2.imread(image_path, 0)
    res = cv2.resize(img, dsize=(32, 48), interpolation=cv2.INTER_LINEAR)
    res = res.reshape(48, 32, 1)
    a1 = res[::-1]
    Image.append(res)
    Image.append(a1)
    label.append([0, 0, 1, 0])
    label.append([0, 0, 0, 1])

label=np.array(label)
Image=np.array(Image)
print('image down')
print(Image.shape)
# -----分类---------------------------------
def next_batch(batch_size):
    x1=[]
    y1=[]
    rs = random.sample(range(0, 2642), batch_size)
    for r in rs:
        x1.append(Image[r])
        y1.append(label[r])

    return x1,y1

'''
sess=tf.Session()
saver = tf.train.import_meta_graph(".//PNEUMONIA.meta")
saver.restore(sess,tf.train.latest_checkpoint("C:/Users/Administrator.DESKTOP-2T54O5R/PycharmProjects/untitled"))

graph = tf.get_default_graph()
ww1=graph.get_tensor_by_name("ww1:0")
ww2=graph.get_tensor_by_name("ww2:0")
ww12=graph.get_tensor_by_name("ww12:0")
ww22=graph.get_tensor_by_name("ww22:0")
wx1=graph.get_tensor_by_name("wx1:0")
wx2=graph.get_tensor_by_name("wx2:0")
wx12=graph.get_tensor_by_name("wx12:0")
wx22=graph.get_tensor_by_name("wx22:0")
bb1=graph.get_tensor_by_name("bb1:0")
bb2=graph.get_tensor_by_name("bb2:0")
bb12=graph.get_tensor_by_name("bb12:0")
bb22=graph.get_tensor_by_name("bb22:0")
bx1=graph.get_tensor_by_name("bx1:0")
bx2=graph.get_tensor_by_name("bx2:0")
bx12=graph.get_tensor_by_name("bx12:0")
bx22=graph.get_tensor_by_name("bx22:0")
'''

batch_size = 64
n_batch = 2642 // batch_size

x = tf.placeholder(tf.float32, [None, 48, 32, 1])
y = tf.placeholder(tf.float32, [None, 4])

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

h_flat = tf.reshape(h_pool2,[-1,3 * 2 * 128])
w_full1 = tf.Variable(tf.truncated_normal([ 3 * 2 * 128,128], stddev=0.1),name='w_full1')
b_full1 = tf.Variable(tf.constant(0.1, shape= [128]),name='b_full1')
h_full1 = tf.nn.relu(tf.matmul(h_flat, w_full1) + b_full1)

keep_pro = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_full1,keep_pro)

w_full2 = tf.Variable(tf.truncated_normal([128, 4], stddev=0.1),name='w_full2') # 人就改成5
b_full2 = tf.Variable(tf.constant(0.1, shape=[4]),name='b_full2')
y_pred = tf.nn.softmax(tf.matmul(h_fc1_drop, w_full2) + b_full2)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
train = tf.train.AdamOptimizer(learning_rate=0.0001,name ='MyNewAdam').minimize(loss)
correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(30):
        count=0
        for batch in range(n_batch):
            count = count+1
            batch_x, batch_y = next_batch(batch_size)
            sess.run(train, feed_dict={x: batch_x, y: batch_y, keep_pro: 0.5})
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_pro: 0.5})
            print(str(count) + " acc: " + str(acc))
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_pro: 0.5})
        print("迭代" + str(epoch+1) + " acc: " + str(acc))

    saver = tf.train.Saver()
    saver.save(sess, ".//self2")




