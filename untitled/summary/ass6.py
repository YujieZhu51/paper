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
people = ['A', 'B', 'C', 'D', 'E']
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
    listVal = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    number = int(filename[0])
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
    listVal1 = [0, 0, 0, 0, 0]
    listVal1[label] = 1
    trainLabelList2.append(listVal1)

for filename in images2List:
    testList.append(os.path.join(path, filename))
    listVal = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    number = int(filename[0])
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
    listVal1 = [0, 0, 0, 0, 0]
    listVal1[label] = 1
    testLabelList2.append(listVal1)
testLabel1 = tf.convert_to_tensor(testLabelList1, dtype=tf.float32)
testLabel2 = tf.convert_to_tensor(testLabelList2, dtype=tf.float32)
trainLabel1 = tf.convert_to_tensor(trainLabelList1, dtype=tf.float32)
trainLabel2 = tf.convert_to_tensor(trainLabelList2, dtype=tf.float32)

testImage = []
trainImage = []

for image_path in trainList:
    image_raw_data = tf.gfile.FastGFile(image_path, 'rb').read()  # ‘r’：UTF-8编码； ‘rb’：非UTF-8编码
    image_data = tf.image.decode_jpeg(image_raw_data)
    trainImage.append(image_data)
trainImage = tf.cast(trainImage, dtype=tf.float32)
trainImage = tf.divide(trainImage, 255)

for image_path in testList:
    image_raw_data = tf.gfile.FastGFile(image_path, 'rb').read()  # ‘r’：UTF-8编码； ‘rb’：非UTF-8编码
    image_data = tf.image.decode_jpeg(image_raw_data)
    testImage.append(image_data)
testImage = tf.cast(testImage, dtype=tf.float32)
testImage = tf.divide(testImage, 255)


# ----------------------读取数据---------------------------------------------------------------

# batch提取方法
def next_batch1(batch_size):
    x1 = []
    y1 = []
    rs = random.sample(range(0, 2700), batch_size)
    for r in rs:
        x1.append(trainImage[r])
        y1.append(trainLabel1[r])
    x1 = tf.convert_to_tensor(x1, dtype=tf.float32)
    y1 = tf.convert_to_tensor(y1, dtype=tf.float32)
    x1 = tf.reshape(x1, [-1, 28, 28])
    x1 = x1.eval()
    y1 = y1.eval()

    return x1, y1

def test_batch1(batch_size):
    x3 = []
    y3 = []
    rs = random.sample(range(0, 300), batch_size)
    for r in rs:
        x3.append(testImage[r])
        y3.append(testLabel1[r])
    x3 = tf.convert_to_tensor(x3, dtype=tf.float32)
    y3 = tf.convert_to_tensor(y3, dtype=tf.float32)
    x3 = tf.reshape(x3, [-1, 28, 28])
    x3 = x3.eval()
    y3 = y3.eval()

    return x3, y3


def next_batch2(batch_size):
    x2 = []
    y2 = []
    rs = random.sample(range(0, 2700), batch_size)
    for r in rs:
        x2.append(trainImage[r])
        y2.append(trainLabel2[r])
    x2 = tf.convert_to_tensor(x2, dtype=tf.float32)
    y2 = tf.convert_to_tensor(y2, dtype=tf.float32)
    x2 = tf.reshape(x2, [-1, 28, 28])
    x2 = x2.eval()
    y2 = y2.eval()

    return x2, y2

# ------------------------------------------------------------------
batch_size = 100
n_batch = 2700 // batch_size
lstm_size = 256 #cell的神经元数,每个时间步长的LSTM单元的数量。


x = tf.placeholder(tf.float32, [None, 28, 28])
y = tf.placeholder(tf.float32, [None, 10])

weights = tf.Variable(tf.truncated_normal([lstm_size,10],stddev = 0.1))
biases = tf.Variable(tf.constant(0.1, shape=[10]))

lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
init_state = lstm_cell.zero_state(batch_size,dtype = tf.float32)
outputs,final_state = tf.nn.dynamic_rnn(lstm_cell,x,initial_state=init_state,time_major = False)# output 每个cell会有一个输出
# 如果为True, 张量的形状为 [max_time, batch_size,cell_size]。如果为False, tensor的形状为[batch_size, max_time, cell_size]
results = tf.nn.softmax(tf.matmul(final_state[1],weights) + biases ) # final_state的形状为 [batch_size, cell.output_size]
# 其中state第一部分为c，代表cell state;第二部分为h，代表hidden state





keep_pro = 0.7
results = tf.nn.dropout(results, keep_pro)


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=results))
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct = tf.equal(tf.argmax(y, 1), tf.argmax(results, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(10):
        count = 0
        for batch in range(n_batch):
            count = count + 1
            print(count)
            batch_x, batch_y = next_batch1(batch_size)
            sess.run(train, feed_dict={x: batch_x, y: batch_y})
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        print("迭代" + str(epoch + 1) + " acc: " + str(acc))
    test_x, test_y = next_batch1(batch_size)
    acc2 = sess.run(accuracy, feed_dict={x: test_x, y: test_y})  # test要全部
    print("test" + " acc: " + str(acc2))