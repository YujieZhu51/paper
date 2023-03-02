import tensorflow as tf
import os
import random

tf.enable_eager_execution()
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
        train_label.append(1)
    if "virus" in filename:
        train_label.append(2)
for filename in train_nor_file_name:
    train_image_path.append(os.path.join(train_nor_path, filename))
    train_label.append(0)

train_label = tf.convert_to_tensor(train_label, dtype=tf.float32)

trainImage = []

for image_path in train_image_path:
    image_raw_data = tf.gfile.FastGFile(image_path, 'rb').read() #‘r’：UTF-8编码； ‘rb’：非UTF-8编码
    image_data = tf.image.decode_jpeg(image_raw_data,channels=1)
    image_data = tf.image.resize_images(image_data, [96,128], 0)
    # 一共有三个参数，第一个是原始图像，第二个是重采样目标影像大小，第三个是重采样的方法。双线性插值算法（Bilinear interpolation）;Method取值为：0；最近邻居法（Nearest  neighbor interpolation);Method取值为：1；双三次插值法（Bicubic interpolation);Method取值为：2
    trainImage.append(image_data)
trainImage = tf.cast(trainImage,dtype=tf.float32)
print(trainImage.shape)