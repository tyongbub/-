import pandas as pd
import matplotlib.pyplot as plt

# 读取xlsx文件
df = pd.read_excel('点坐标.xlsx')

# 获取前500行数据
df_head = df.head(9000)

# 使用行号作为横坐标
x_values = df_head['X坐标']

# 纵坐标为'Singular Value 288'列
y_values = df_head['Y坐标']

# 绘制点图
plt.plot(x_values, y_values, marker='o', linestyle='None', color='blue', markersize=5)
#plt.scatter(x_values, y_values, marker='o', facecolors='none', edgecolors='blue')
plt.xlabel('Row Number')
plt.ylabel('Singular Value 288')
plt.title('Scatter Plot of Singular Value 288 with Row Number')
plt.xticks(range(0, 2000, 200))
plt.grid(True)
plt.show()




