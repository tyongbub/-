import tensorflow as tf
import os

from sklearn.metrics import silhouette_score

os.environ["OMP_NUM_THREADS"] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
from scipy.linalg import svd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers, models
import matplotlib.pyplot as plt
import cv2
import numpy as np


dataset_folder = 'gasimagevideo/video2'

# 加载数据集
def load_dataset(folder):
    images = []
    # 遍历文件夹中的每个图像文件
    for filename in os.listdir(folder):
        # 只处理图像文件
        if filename.endswith(('.bmp')):
            # 使用 OpenCV 加载图像
            img = cv2.imread(os.path.join(folder, filename))
            # 将图像转换为灰度图像
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 将图像调整为统一的大小
            # img = cv2.resize(img, (28, 28))
            # 将图像添加到列表中
            images.append(img)
    return np.array(images)
'''
# 定义卷积神经网络
def create_cnn(input_shape):
    model = models.Sequential([
        layers.Reshape(input_shape + (1,)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Flatten(),
        layers.Dense(128, activation='relu')
    ])
    return model
'''
# 定义自编码器模型
class AE(keras.Model):
    def __init__(self):
        super(AE, self).__init__()

        # 编码器
        self.encoder = keras.Sequential([
            layers.Input(shape=(288, 384, 1)),  # 输入数据的维度
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu')  # 降维到32维特征
        ])

        # 解码器
        self.decoder = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(288 * 384 * 1, activation='sigmoid'),  # 输出维度调整为输入维度
            layers.Reshape((288, 384, 1))  # 调整为与输入数据相同的维度
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

# 加载整个数据集
data = load_dataset(dataset_folder)
# 加载训练集和测试集
x_train, x_test = train_test_split(data, test_size=0.2, random_state=42)


# 对图像进行预处理
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# 创建自编码器模型实例
model = AE()

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy')

# 训练自编码器
model.fit(x_train, x_train, epochs=10, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

# 使用编码器提取特征值
encoded_imgs = model.encoder(x_test).numpy()

# 加载图像数据集并计算奇异值
singular_values = []

# 遍历文件夹中的每个图像文件
for filename in os.listdir(dataset_folder):
    # 只处理图像文件
    if filename.endswith('.bmp'):
        # 使用 OpenCV 加载图像并转换为灰度图像
        img = cv2.imread(os.path.join(dataset_folder, filename), cv2.IMREAD_GRAYSCALE)
        # 计算图像的奇异值分解
        U, s, Vt = svd(img)
        # 将奇异值存储在列表中
        singular_values.append(s)

# 执行K均值聚类
from sklearn.cluster import KMeans

num_clusters = 3  # 修改为3类

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(singular_values)  # 使用奇异值进行聚类

# 计算聚类中心
cluster_centers = kmeans.cluster_centers_

# 计算聚类中心之间的欧氏距离
from scipy.spatial.distance import euclidean
from itertools import combinations

center_distances = []

distance = euclidean(cluster_centers[0], cluster_centers[1])
center_distances.append(distance)
print(distance)

distance = euclidean(cluster_centers[0], cluster_centers[2])
center_distances.append(distance)
print(distance)

distance = euclidean(cluster_centers[1], cluster_centers[2])
center_distances.append(distance)
print(distance)