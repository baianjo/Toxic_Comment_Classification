import pandas as pd                                             # 数据处理库 - 读取CSV、清洗数据、统计分析的瑞士军刀
from sklearn.model_selection import train_test_split            # 数据划分工具 - 把数据分成训练集和测试集，让模型先学习再考试
from sklearn.feature_extraction.text import TfidfVectorizer     # 文本向量化工具 - 把文字转成数字，分析词汇重要程度
from sklearn.linear_model import LogisticRegression             # 逻辑回归模型 - AI分类器，判断评论是否有毒
from sklearn.metrics import classification_report               # 模型评估工具 - 生成准确率、召回率等性能报告
from scipy.sparse import csr_matrix                             # 稀疏矩阵类型注解

print("--- 步骤1：加载数据 ---")
df = pd.read_csv('train.csv')
print("数据加载完毕。")

# 为了让模型跑得快一点，我们先只用前30000行数据做个快速实验
df_sample = df.head(30000)
print(f"本次实验使用 {len(df_sample)} 行数据。")


# 目标是先只预测toxic这一个标签
print("--- 步骤2：准备数据和标签，并划分训练集和测试集 ---")
# 定义"问题"和"答案" (X 和 y)
# 在《练习册》和《模拟考卷》里，每一页都包含两部分：
# X (The Problem): comment_text列，这是问题。比如："你真是个天才！"
# y (The Answer): toxic列（里面的0或1），这是标准答案。比如：0 (代表"不是坏话")。
# X是我们的"情报"（评论文本）
X = df_sample['comment_text']  # 为了让机器学习算法能够分析文本，需要明确指定输入特征
# y是我们的“标签”（是否 toxic）
y = df_sample['toxic']  # 为了训练监督学习模型，需要提供标准答案作为学习目标

# 使用train_test_split函数自动划分训练集和测试集
# test_size=0.3 意味着我们把30%的数据作为“模拟考试”，70%用于学习
# random_state=42 是一个随机数种子，保证每次运行时，划分的训练集和测试集都是相同的，便于复现
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# X_train 是《练习册》里的所有问题。
# y_train 是《练习册》里所有问题的标准答案。
# X_test 是《模拟考卷》里的所有问题。
# y_test 是《模拟考卷》里所有问题的标准答案（这份答案，学生在考试结束前绝对不能看）

print(f"训练集大小：{len(X_train)}")
print(f"测试集大小：{len(X_test)}")


print("--- 步骤3：文本向量化（将文字转化为数字）---")
# 机器学习模型是个超级偏科的数学天才，他不认识字，只认识数字。
# 所以，我们需要一个"加密"工具，把X_train和X_test里的所有句子，都变成一长串数字（我们称之为"向量"）。
# TfidfVectorizer就是这个加密工具

# 初始化TF-IDF向量化器
# max_features=5000 表示只考虑前5000个最常用的单词
vectorizer = TfidfVectorizer(max_features=5000)

# 让向量化机器学习训练集（X_train）的词汇，并把它转换成数字矩阵
X_train_tfidf = vectorizer.fit_transform(X_train)  # type: ignore

# 使用同一个向量化器（同一本密码本）来转换测试集（X_test）
X_test_tfidf = vectorizer.transform(X_test)  # type: ignore

print("TF-IDF向量化完成！训练数据的数字矩阵形态：", X_train_tfidf.shape)  # type: ignore


print("--- 步骤4：训练逻辑回归模型---")
model = LogisticRegression()

model.fit(X_train_tfidf, y_train)

print("模型训练完成！")


print("--- 步骤5：在测试集上进行预测---")
# 使用.predict()方法，让训练好的模型对“模拟考试”数据进行预测
y_pred = model.predict(X_test_tfidf)

print("预测完成！")


print("--- 步骤6：评估模型性能---")

# 用classification_report()函数生成一份详细的战报
# 它会比较模型的预测（y_pred）和真实值（y_test）
report = classification_report(y_test, y_pred)

print("战报：")
print(report)