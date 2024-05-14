import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 定义函数来计算奇异值平均值
def calculate_svd_mean(folder):
    file_list = os.listdir(folder)
    svd_means = []
    for file_name in file_list:
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            image_path = os.path.join(folder, file_name)
            # 加载图像
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Error: Unable to load image {image_path}")
                continue

            # 图像增强：应用高斯模糊和直方图均衡化
            image = cv2.GaussianBlur(image, (5, 5), 0)
            image = cv2.equalizeHist(image)
            # 计算奇异值
            _, s, _ = np.linalg.svd(image)
            svd_mean = np.mean(s)
            svd_means.append(svd_mean)
    return np.mean(svd_means)

# 定义文件夹列表
base_folder = r'F:\AutoEncoder-KMeans-master\cluster_results'
folders = [os.path.join(base_folder, str(i)) for i in range(10)]

# 计算每个文件夹的奇异值平均值
svd_means = []
for folder in folders:
    svd_mean = calculate_svd_mean(folder)
    if not np.isnan(svd_mean):  # 检查结果是否为NaN
        svd_means.append(svd_mean)

# 绘制图表
plt.figure(figsize=(10, 6))
plt.bar(range(len(svd_means)), svd_means, color='skyblue')
plt.xlabel('Folder')
plt.ylabel('Average Singular Value')
plt.title('Average Singular Value of Images in Each Folder (with Image Enhancement)')
plt.xticks(range(len(svd_means)), range(len(svd_means)))
plt.ylim(340, 350)  # 设置纵坐标范围为340到350
plt.show()


