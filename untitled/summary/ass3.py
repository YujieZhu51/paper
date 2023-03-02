import tensorflow as tf
import os
import random

path = "D://deep_learning//image//image//"

imagesList = []  # 存放图片数据路径的列表
trainList = []
images2List = []
testLabelList1 = []
testLabelList2 = []  # 存放标签，与上面图片是一一对应的
trainLabelList1 = []
trainLabelList2 = []
for filename in os.listdir(path):  # 获取图片的名字
    imagesList.append(filename)

# 分出test
testList = []
people = ['A','B','C','D','E']
for person in people:
    for num in range(10):
        for times in range(6):
            for p in imagesList:
                if p[-5] == person:
                    if int(p[0]) == num:
                        images2List.append(p)
                        imagesList.remove(p)
                        break
# 提label

for filename in imagesList:
    trainList.append(os.path.join(path, filename))
    listVal = [0,0,0,0,0,0,0,0,0,0]
    number =int(filename[0])
    listVal[number] = 1
    trainLabelList1.append(listVal)
    name = filename[-5]
    label = 5
    if name == 'A':
        label = 0
    elif name == 'B':
        label = 1
    elif name == 'C':
        label = 2
    elif name == 'D':
        label = 3
    elif name == 'E':
        label = 4
    listVal1 = [0,0,0,0,0]
    listVal1[label] = 1
    trainLabelList2.append(listVal1)




for filename in images2List:
    testList.append(os.path.join(path, filename))
    listVal = [0,0,0,0,0,0,0,0,0,0]
    number =int(filename[0])
    listVal[number] = 1
    testLabelList1.append(listVal)
    name = filename[-5]
    label = 5
    if name == 'A':
        label = 0
    elif name == 'B':
        label = 1
    elif name == 'C':
        label = 2
    elif name == 'D':
        label = 3
    elif name == 'E':
        label = 4
    listVal1 = [0,0,0,0,0]
    listVal1[label] = 1
    testLabelList2.append(listVal1)
testLabel1 = tf.convert_to_tensor(testLabelList1, dtype=tf.float32)
testLabel2 = tf.convert_to_tensor(testLabelList2, dtype=tf.float32)
trainLabel1 = tf.convert_to_tensor(trainLabelList1, dtype=tf.float32)
trainLabel2 = tf.convert_to_tensor(trainLabelList2, dtype=tf.float32)




testImage = []
trainImage = []

for image_path in trainList:
    image_raw_data = tf.gfile.FastGFile(image_path, 'rb').read() #‘r’：UTF-8编码； ‘rb’：非UTF-8编码
    image_data = tf.image.decode_jpeg(image_raw_data)
    trainImage.append(image_data)
trainImage = tf.cast(trainImage,dtype=tf.float32)
trainImage = tf.divide(trainImage,255)

for image_path in testList:
    image_raw_data = tf.gfile.FastGFile(image_path, 'rb').read() #‘r’：UTF-8编码； ‘rb’：非UTF-8编码
    image_data = tf.image.decode_jpeg(image_raw_data)
    testImage.append(image_data)
testImage = tf.cast(testImage,dtype=tf.float32)
testImage = tf.divide(testImage,255)







# ----------------------读取数据---------------------------------------------------------------

# batch提取方法
def next_batch1(batch_size):
    x1=[]
    y1=[]
    rs = random.sample(range(0, 2700), batch_size)
    for r in rs:
        x1.append(trainImage[r])
        y1.append(trainLabel1[r])
    x1 = tf.convert_to_tensor(x1, dtype=tf.float32)
    y1 = tf.convert_to_tensor(y1, dtype=tf.float32)
    x1=x1.eval()
    y1=y1.eval()

    return x1,y1

def next_batch2(batch_size):
    x2=[]
    y2=[]
    rs = random.sample(range(0, 2700), batch_size)
    for r in rs:
        x2.append(trainImage[r])
        y2.append(trainLabel2[r])
    x2 = tf.convert_to_tensor(x2, dtype=tf.float32)
    y2 = tf.convert_to_tensor(y2, dtype=tf.float32)
    x2=x2.eval()
    y2=y2.eval()

    return x2,y2



    
batch_size = 300
n_batch =2700 // batch_size


x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 5])

# ----------CNN第一层-----------------------------
w1 = tf.Variable(tf.truncated_normal([3,3,1,16], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape= [16]))
c1 = tf.nn.conv2d(x,w1,strides=[1,1,1,1],padding='SAME') + b1
h_conv1 = tf.nn.relu(c1)
# ----------CNN第二层-----------------------------

w2 = tf.Variable(tf.truncated_normal([3,3,16,32], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape= [32]))
c2 = tf.nn.conv2d(h_conv1,w2,strides=[1,1,1,1],padding='SAME') + b2
h_conv2 = tf.nn.relu(c2)

# ----------池化-----------------------------
h_pool1 = tf.nn.max_pool(h_conv2,ksize = [1,2,2,1],strides=[1,2,2,1],padding='SAME')


# ----------CNN第三层-----------------------------
w3 = tf.Variable(tf.truncated_normal([3,3,32,64], stddev=0.1))
b3 = tf.Variable(tf.constant(0.1, shape= [64]))
c3 = tf.nn.conv2d(h_pool1,w3,strides=[1,1,1,1],padding='SAME') + b3
h_conv3 = tf.nn.relu(c3)
# ----------CNN第四层-----------------------------

w4 = tf.Variable(tf.truncated_normal([3,3,64,64], stddev=0.1))
b4 = tf.Variable(tf.constant(0.1, shape= [64]))
c4 = tf.nn.conv2d(h_conv3,w4,strides=[1,1,1,1],padding='SAME') + b4
h_conv4 = tf.nn.relu(c4)

# ----------池化-----------------------------
h_pool2 = tf.nn.max_pool(h_conv4,ksize = [1,2,2,1],strides=[1,2,2,1],padding='SAME')


# ----------全连接-----------------------------
h_flat = tf.reshape(h_pool2,[-1,7 * 7 * 64])
w_full1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 256], stddev=0.1))
b_full1 = tf.Variable(tf.constant(0.1, shape= [256]))
h_full1 = tf.nn.relu(tf.matmul(h_flat, w_full1) + b_full1)

keep_pro = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_full1,keep_pro)

w_full2 = tf.Variable(tf.truncated_normal([256, 5], stddev=0.1)) # 人就改成5
b_full2 = tf.Variable(tf.constant(0.1, shape=[5]))

y_pred = tf.nn.softmax(tf.matmul(h_fc1_drop, w_full2) + b_full2)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))# ?为什么是y不是全部的list
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(7):
        count=0
        for batch in range(n_batch):
            count = count+1
            print(count)
            batch_x, batch_y = next_batch2(batch_size)
            sess.run(train, feed_dict={x: batch_x, y: batch_y, keep_pro: 0.7})
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_pro: 0.7})
        print("迭代" + str(epoch+1) + " acc: " + str(acc))
    acc2 = sess.run(accuracy, feed_dict={x: testImage.eval(), y: testLabel2.eval(), keep_pro: 1}) # test要全部
    print("test" + " acc: " + str(acc2))

7



