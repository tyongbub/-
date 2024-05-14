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

dataset_folder = 'gasimagevideo/video1'

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

# 定义变分自编码器模型
class VAE(keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = keras.Sequential([
            layers.Input(shape=(288, 384, 1)),  # 输入数据的维度
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(latent_dim + latent_dim)  # 输出均值和方差
        ])
        self.decoder = keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(288 * 384 * 1, activation='sigmoid'),  # 输出维度调整为输入维度
            layers.Reshape((288, 384, 1))  # 调整为与输入数据相同的维度
        ])

    def sample(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs):
        z_mean, z_log_var = tf.split(self.encoder(inputs), num_or_size_splits=2, axis=1)
        z = self.sample(z_mean, z_log_var)
        decoded = self.decoder(z)
        return decoded

# 加载整个数据集
data = load_dataset(dataset_folder)
# 加载训练集和测试集
x_train, x_test = train_test_split(data, test_size=0.2, random_state=42)

# 对图像进行预处理
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# 创建VAE模型实例
latent_dim = 32  # 潜在空间的维度
model = VAE(latent_dim)

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy')

# 训练VAE
model.fit(x_train, x_train, epochs=10, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

# 生成新的图像数据
def generate_images(model, num_images=10, latent_dim=32):
    # 在潜在空间中采样
    latent_vectors = np.random.normal(size=(num_images, latent_dim))
    # 解码器生成图像
    generated_images = model.decoder.predict(latent_vectors)
    return generated_images

# 选择要生成的图像数量
num_images_to_generate = 10

# 使用模型生成新的图像数据
generated_images = generate_images(model, num_images=num_images_to_generate)

# 可视化生成的图像
plt.figure(figsize=(10, 4))
for i in range(num_images_to_generate):
    plt.subplot(2, num_images_to_generate // 2, i + 1)
    plt.imshow(generated_images[i].reshape(288, 384), cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()

# 使用编码器提取特征值
encoded_imgs = model.encoder(x_test).numpy()


# 执行K均值聚类
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10, random_state=42,n_init='auto')
clusters = kmeans.fit_predict(encoded_imgs)

# 计算轮廓系数
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(encoded_imgs)
    silhouette_avg = silhouette_score(encoded_imgs, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# 绘制折线图
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.grid(True)
plt.show()

print(clusters)

save_dir = 'cluster_results'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 绘制并保存每个簇的图像
for i in range(10):  # 假设有10个聚类簇
    cluster_samples = x_test[clusters == i]  # 获取属于第i个簇的样本
    for j, sample in enumerate(cluster_samples):
        cluster_dir = os.path.join(save_dir, f'{i}')
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)
        plt.imshow(sample, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(cluster_dir, f'image_{j}.png'))
        plt.close()

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

# 将奇异值转换为 DataFrame
singular_df = pd.DataFrame(singular_values)

# 制作表格并填入奇异值
singular_df.index = [f"Image {i+1}" for i in range(len(singular_df))]
singular_df.columns = [f"Singular Value {i+1}" for i in range(len(singular_df.columns))]

# 打印 DataFrame
#print(singular_df)

# 导出到 Excel 文件
excel_filename = 'singular_values.xlsx'
singular_df.to_excel(excel_filename, index=True)
print(f"DataFrame 已成功导出到 Excel 文件: {excel_filename}")


