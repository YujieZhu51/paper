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
p4='C://deep_learning//COVID DATA//COVID DATA//CT_COVID//CT_COVID'
p5='C://deep_learning//COVID DATA//COVID DATA//CT_NonCOVID'
filename0=[]
filename1=[]
filename2=[]
filename3=[]
image_path_cov=[]
image_path_non=[]
for filename in os.listdir(p0):
    filename0.append(filename)
for filename in os.listdir(p1):
    filename1.append(filename)
for filename in os.listdir(p2):
    filename2.append(filename)
for filename in os.listdir(p3):
    filename3.append(filename)

for filename in filename0:
    image_path_non.append(os.path.join(p0, filename))
for filename in filename1:
    image_path_non.append(os.path.join(p1, filename))
for filename in filename2:
    image_path_cov.append(os.path.join(p2, filename))
for filename in filename3:
    image_path_cov.append(os.path.join(p3, filename))




Image = []
label = []
trainImage = []
trainlabel = []
testImage = []
testlabel = []


for image_path in image_path_non:
        img = cv2.imread(image_path, 0)
        res = cv2.resize(img, dsize=(32, 48), interpolation=cv2.INTER_LINEAR)
        a1 = cv2.flip(res, 1)
        res = res.reshape(48, 32, 1)
        a1 = a1.reshape(48, 32, 1)
        Image.append(res)
        Image.append(a1)
        label.append([1, 0])
        label.append([1, 0])
for image_path in image_path_cov:
        img = cv2.imread(image_path, 0)
        res = cv2.resize(img, dsize=(32, 48), interpolation=cv2.INTER_LINEAR)
        a1 = cv2.flip(res, 1)
        res = res.reshape(48, 32, 1)
        a1 = a1.reshape(48, 32, 1)
        Image.append(res)
        Image.append(a1)
        label.append([0, 1])
        label.append([0, 1])

rs = random.sample(range(0, 2642), 400)

for i in range(0,2642):
    whether = rs.count(i)
    if whether == 0:
        trainImage.append(Image[i])
        trainlabel.append(label[i])
    else:
        testImage.append(Image[i])
        testlabel.append(label[i])
trainlabel=np.array(trainlabel)
trainImage=np.array(trainImage)
testlabel=np.array(testlabel)
testImage=np.array(testImage)

print(testImage.shape)
print(trainImage.shape)
print('image down')
# ---------两叶肺----------------------------
filename4=[]
filename5=[]
image_p_c=[]
image_p_n=[]
for filename in os.listdir(p4):
    filename4.append(filename)
for filename in os.listdir(p5):
    filename5.append(filename)

for filename in filename5:
    image_p_n.append(os.path.join(p5, filename))
for filename in filename4:
    image_p_c.append(os.path.join(p4, filename))

allImage = []
allLabel = []

rs = random.sample(range(0, 746), 146)
num=0
for image_path in image_p_n:
    whether = rs.count(num)
    if whether == 1:
        img = cv2.imread(image_path, 0)
        res = cv2.resize(img, dsize=(32, 48), interpolation=cv2.INTER_LINEAR)
        res = res.reshape(48, 32, 1)
        allImage.append(res)
        allLabel.append([1,0])
    num=num+1

for image_path in image_p_c:
    whether = rs.count(num)
    if whether == 1:
        img = cv2.imread(image_path, 0)
        res = cv2.resize(img, dsize=(32, 48), interpolation=cv2.INTER_LINEAR)
        res = res.reshape(48, 32, 1)
        allImage.append(res)
        allLabel.append([0,1])
    num=num+1

allLabel=np.array(allLabel)
allImage=np.array(allImage)
print(allImage.shape)

# -----分类---------------------------------
def next_batch(batch_size):
    x1=[]
    y1=[]
    rs = random.sample(range(0, 2242), batch_size)
    for r in rs:
        x1.append(trainImage[r])
        y1.append(trainlabel[r])
    #x1 = tf.convert_to_tensor(x1, dtype=tf.float32)
    #y1 = tf.convert_to_tensor(y1, dtype=tf.float32)
    #x1=x1.eval()
    #y1=y1.eval()

    return x1,y1

sess=tf.Session()
saver = tf.train.import_meta_graph(".//self2.meta")#(".//PNEUMONIA.meta")
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



batch_size = 64
n_batch = 2242 // batch_size

x = tf.placeholder(tf.float32, [None, 48, 32, 1])
y = tf.placeholder(tf.float32, [None, 2])

#ww1 = tf.Variable(tf.truncated_normal([3,3,1,16], stddev=0.1),name='ww1')
#bb1 = tf.Variable(tf.constant(0.1, shape= [16]),name='bb1')
c1 = tf.nn.conv2d(x,ww1,strides=[1,1,1,1],padding='SAME') + bb1
#wx1 = tf.Variable(tf.truncated_normal([1,1,1,16], stddev=0.1),name='wx1')
#bx1 = tf.Variable(tf.constant(0.1, shape= [16]),name='bx1')
x1 = tf.nn.conv2d(x,wx1,strides=[1,1,1,1],padding='SAME') + bx1
add1 = tf.add(c1, x1)
nor1=tf.layers.batch_normalization(add1,training=True)
h_conv1 = tf.nn.relu(nor1)

#ww12 = tf.Variable(tf.truncated_normal([3,3,16,32], stddev=0.1),name='ww12')
#bb12 = tf.Variable(tf.constant(0.1, shape= [32]),name='bb12')
c12 = tf.nn.conv2d(h_conv1,ww12,strides=[1,1,1,1],padding='SAME') + bb12
#wx12 = tf.Variable(tf.truncated_normal([1,1,16,32], stddev=0.1),name='wx12')
#bx12 = tf.Variable(tf.constant(0.1, shape= [32]),name='bx12')
x12 = tf.nn.conv2d(h_conv1,wx12,strides=[1,1,1,1],padding='SAME') + bx12
add12 = tf.add(c12, x12)
nor12=tf.layers.batch_normalization(add12,training=True)
h_conv12 = tf.nn.relu(nor12)


h_pool1 = tf.nn.max_pool(h_conv12,ksize = [1,4,4,1],strides=[1,4,4,1],padding='SAME')

#ww2 = tf.Variable(tf.truncated_normal([3,3,32,64], stddev=0.1),name='ww2')
#bb2 = tf.Variable(tf.constant(0.1, shape= [64]),name='bb2')
c2 = tf.nn.conv2d(h_pool1,ww2,strides=[1,1,1,1],padding='SAME') + bb2
#wx2 = tf.Variable(tf.truncated_normal([1,1,32,64], stddev=0.1),name='wx2')
#bx2 = tf.Variable(tf.constant(0.1, shape= [64]),name='bx2')
x2 = tf.nn.conv2d(h_pool1,wx2,strides=[1,1,1,1],padding='SAME') + bx2
add2 = tf.add(c2, x2)
nor2=tf.layers.batch_normalization(add2,training=True)
h_conv2 = tf.nn.relu(nor2)

#ww22 = tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.1),name='ww22')
#bb22 = tf.Variable(tf.constant(0.1, shape= [128]),name='bb22')
c22 = tf.nn.conv2d(h_conv2,ww22,strides=[1,1,1,1],padding='SAME') + bb22
#wx22 = tf.Variable(tf.truncated_normal([1,1,64,128], stddev=0.1),name='wx22')
#bx22 = tf.Variable(tf.constant(0.1, shape= [128]),name='bx22')
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

w_full2 = tf.Variable(tf.truncated_normal([128, 2], stddev=0.1),name='w_full2') # 人就改成5
b_full2 = tf.Variable(tf.constant(0.1, shape=[2]),name='b_full2')
y_pred = tf.nn.softmax(tf.matmul(h_fc1_drop, w_full2) + b_full2)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
train = tf.train.AdamOptimizer(learning_rate=0.0001,name ='MyNewAdam2').minimize(loss)
correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(26):
        count=0
        for batch in range(n_batch):
            count = count+1
            batch_x, batch_y = next_batch(batch_size)
            sess.run(train, feed_dict={x: batch_x, y: batch_y, keep_pro: 0.5})
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_pro: 0.5})
            print(str(count) + " acc: " + str(acc))
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_pro: 0.5})
        print("迭代" + str(epoch+1) + " acc: " + str(acc))
    acc2 = sess.run(accuracy, feed_dict={x: testImage, y: testlabel, keep_pro: 1})
    print("test" + " acc: " + str(acc2))
    acc3 = sess.run(accuracy, feed_dict={x: allImage, y: allLabel, keep_pro: 1})
    print("test2" + " acc: " + str(acc3))



