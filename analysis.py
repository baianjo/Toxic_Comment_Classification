import pandas as pd                                 # 数据处理库
import numpy as np                                  # 数值计算库
import matplotlib.pyplot as plt                     # 基础绘图库
import seaborn as sns                               # 高级绘图库
from matplotlib.font_manager import FontProperties  # 字体配置工具


# 使用微软雅黑字体文件作为主要字体
my_font = FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc')  # 微软雅黑字体文件
print("所有工具已准备就绪！")


file_path = 'train.csv'
df = pd.read_csv(file_path)

# 显示数据的前5行，看看长什么样
print("\n数据前5行预览：")
print(df.head())

# 显示数据集的摘要信息
print("\n数据体检报告：")
print(df.info())

# 显示数字列的统计摘要
print("\n数字列统计摘要：")
print(df.describe())

# 统计各标签的计数
label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
# df[label_columns]: 选择DataFrame中的标签列（有害评论分类列）
# .sum(): 对每列进行求和，由于标签列只包含0和1，求和结果就是每种类型的评论数量
# 返回一个Series，索引为列名，值为对应列的总数
label_counts = df[label_columns].sum()
print("\n各“毒舌”标签的统计：")
print(label_counts)

# 新建一列label_sum 用于保存每行评论的毒性标签的总和
df['label_sum'] = df[label_columns].sum(axis=1)
print("\n数据前5行预览：")
print(df.head())
is_clean = df['label_sum'] == 0                     # 存着类似 [True, True, False, True, ...]，是过滤器、掩码
clean_comments_count = is_clean.sum()
print("\n无毒性评论的个数：", clean_comments_count)

# 找出最大毒性
max_toxicity = df['label_sum'].max()
print("\n最大毒性：", max_toxicity)
# 1 第一个最大毒性的评论
most_toxic_row_first_id = df['label_sum'].idxmax()
print("\n最大毒性的评论索引：", most_toxic_row_first_id)
most_toxic_first_row = df.loc[most_toxic_row_first_id]
most_toxic_first_comment = most_toxic_first_row['comment_text']
print("\n第一个最大毒性的评论：", most_toxic_first_comment)
# 2 全部最大毒性的评论
is_most_toxic = df['label_sum'] == max_toxicity     # 存着类似 [True, True, False, True, ...]，是过滤器、掩码
most_toxic_rows = df[is_most_toxic]
print("\n全部最大毒性的评论前5行预览：")
print(most_toxic_rows.head())



# 设置图表样式
sns.set(style="whitegrid")

# 创建一个图表，figsize = (8, 5) 表示图表的宽度和高度
plt.figure(figsize=(8, 5))

# 使用seaborn 的 barplot 函数绘制柱状图
# x 轴的数据是 label_counts 的索引，即标签名称
# y 轴的数据是 label_counts 的值，即标签的计数
sns.barplot(x=label_counts.index, y=label_counts.values, hue=label_counts.index, palette="viridis", legend=False)

# 设置图表标题
plt.title("各“毒舌”评论的数量分布", fontproperties=my_font, fontsize=16)
plt.xlabel("毒性标签类型", fontproperties=my_font, fontsize=14)
plt.ylabel("评论数量", fontproperties=my_font, fontsize=14)

# 让x轴的标签旋转45度 - 避免标签重叠，显示更清晰
plt.xticks(rotation=45)

# 这就是我们画图时用到的y轴数据，它是一个NumPy数组
y_values = label_counts.values


# 显示图表
plt.show()
