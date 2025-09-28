import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer # 新的、更强大的"密码本"工具
from tensorflow.keras.utils import pad_sequences # 负责把情报"裁剪"成同样长度的工具 - 新版本路径
from tensorflow.keras.models import Sequential # 搭建神经网络的"乐高"盒子
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense # 我们要用的三种"乐高积木"。
from sklearn.metrics import classification_report

# 未使用SEED

print("--- 步骤1：加载并准备数据 ---")
df = pd.read_csv('train.csv')
df_sample = df.head(30000)

X = df_sample['comment_text']
y = df_sample['toxic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("数据准备完毕。")



print("\n--- 步骤2：使用Tokenizer进行文本数字化 ---")
# --- “密码本”制作 ---
# 初始化Tokenizer，num_words=10000 意味着我们的新密码本只收录最重要的10000个词
tokenizer = Tokenizer(num_words=10000)
# 使用 .fit_on_texts() 命令，让tokenizer学习训练集(X_train)的所有词汇
tokenizer.fit_on_texts(X_train)

# --- 使用“密码本”翻译情报 ---
# 将训练集的文本，翻译成由整数“代号”组成的序列
X_train_sequences = tokenizer.texts_to_sequences(X_train)
# 将测试集的文本，也用同一本密码本进行翻译
X_test_sequences = tokenizer.texts_to_sequences(X_test)

print("第一句训练文本的原文:", X_train.iloc[0])  # 展示原始文本内容 - 就像展示一篇中文文章的原文，让我们看看机器要处理的是什么样的评论
print("翻译后的整数序列:", X_train_sequences[0])  # 展示数字化后的结果 - 就像把中文翻译成了数字密码，每个数字代表一个词，机器只能理解数字



print("\n--- 步骤3：使用Padding统一情报长度 ---")
# 神经网络要求所有输入的情报长度必须一致。我们设定一个最大长度，比如200
# 【为什么需要统一长度？】
# 1. 矩阵运算需求：神经网络本质上是矩阵运算，就像Excel表格的每一行必须有相同的列数才能进行计算
# 2. 批量处理效率：计算机需要一次处理多条评论，就像工厂流水线上的产品必须规格统一才能批量加工
# 3. 内存分配要求：计算机需要预先分配固定大小的内存空间
# 4. 权重参数匹配：神经网络的权重矩阵大小是固定的，输入长度变化会导致计算无法进行
maxlen = 200
# 使用 pad_sequences 工具统一序列长度，因为神经网络要求输入数据的形状必须一致
# 长评论截断：保留前200个词，去掉后面的内容（避免信息过载）；短评论填充：用0补齐到200长度（保证矩阵运算正确进行）



X_train_padded = pad_sequences(X_train_sequences, maxlen=200, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=200, padding='post', truncating='post')
# "You are an idiot"

# 原始: ["You", "are", "an", "idiot"]
#   ↓
# 密码本翻译: [10, 2, 8, 897]
#   ↓
# 标准化填充 (假设 maxlen=10): [10, 2, 8, 897, 0, 0, 0, 0, 0, 0]


print("Padding完成。处理后的训练数据形态:", X_train_padded.shape)
print("处理后的第一句训练文本:\n", X_train_padded[0])




print("\n--- 步骤4：搭建神经网络模型 ---")
# 定义一个空的“乐高”盒子
model = Sequential()

# 第一层：嵌入层 (Embedding Layer) - 我们的“声纳”
# 这是魔法发生的地方。它会把每个词的“整数代号”转换成一个16维的“GPS坐标”向量
# input_dim=10000 对应我们密码本的词汇量
# output_dim=16 意味着每个词的GPS坐标有16个维度
# input_length=maxlen 对应我们padding后的标准长度



#          坐标1   坐标2    坐标3  ...  坐标16
#       +--------------------------------------+
# ID 10 |  0.12   -0.45    0.88  ...   -0.23   |  <- "You" 的语义坐标
# ID 2  |  0.05    0.11   -0.02  ...    0.51   |  <- "are" 的语义坐标
# ID 8  |  0.01   -0.03    0.04  ...   -0.09   |  <- "an" 的语义坐标
# ID 897| -0.91    0.85    0.76  ...    0.66   |  <- "idiot" 的语义坐标
# ID 0  |  0.00    0.00    0.00  ...    0.00   |
# ID 0  |  0.00    0.00    0.00  ...    0.00   |
# ...   |  ...     ...     ...   ...    ...    | (后面都是0)
# ID 0  |  0.00    0.00    0.00  ...    0.00   |
#       +--------------------------------------+
model.add(Embedding(input_dim=10000, output_dim=16, input_length=maxlen))



# 第二层：全局平均池化层 (GlobalAveragePooling1D) - “情报汇总官”
# 它会把一句评论（200个词的坐标）的所有词的坐标向量取一个平均值，
# 得到一个能代表整句话语义的、浓缩的16维向量。

# 之前 (10 x 16):
# +-------------------+
# | vector for "You"  |
# | vector for "are"  |
# | ...               |
# +-------------------+
#           |
#           V
# 现在 (1 x 16):
# +------------------------------------------------------+
# | -0.073 | 0.048 | ... | 0.085                         |  <- 代表“You are an idiot”整句话的“平均语义”
# +------------------------------------------------------+
model.add(GlobalAveragePooling1D())




# 第三层：全连接层 (Dense Layer) - “高级分析师”
# 这是神经网络的核心。它会学习如何从那16维的句子向量中发现复杂的模式。
# 24是神经元的数量，可以理解为24个分析师在同时工作。
# activation='relu' 是一种激活函数，像一个开关，只允许重要的信号通过。

#                              +--------------+
#                              | 专家1: 8.5    |  <- "脏话信号很强！"
#                              +--------------+
#                              +--------------+
# +-------------------+        | 专家2: 1.2    |
# | 16维句子总结向量    |  --->  +--------------+
# +-------------------+        | ...          |
#                              +--------------+
#                              | 专家24: 0.0   |  <- "没发现讽刺意味。" (`relu`激活函数会把负分直接变成0)
#                              +--------------+
model.add(Dense(24, activation='relu'))



# 第四层：输出层 (Output Layer) - “最终决策官”
# 最后的决策者。它只有一个神经元，因为我们只需要一个输出：是toxic(1)还是不是(0)。
# activation='sigmoid' 把任何数值都压缩到0-1之间，可以被解读为“是toxic的概率”。

# 大法官的总分: 3.47
#      |
#      V
# +-----------------+
# | sigmoid(3.47)   |
# +-----------------+
#      |
#      V
# 最终概率: 0.97
model.add(Dense(1, activation='sigmoid'))


# 编译模型：给模型设定“学习目标”和“优化方法”
model.compile(optimizer='adam',
              loss='binary_crossentropy', # 告诉模型这是一个二分类问题
              metrics=['accuracy'])

# 打印出我们搭建好的模型结构
model.summary()




print("\n--- 步骤5：训练模型 ---")

# --- VERSION2：由于第一次尝试的recall特别低(<0.1)，所以我们新增代码：计算类别权重 ---
# 我们的目标是让少数类（toxic）的权重更高
# 公式：总样本数 / (类别数 * 该类别样本数)
# 这样可以给样本少的类别一个更高的权重
from sklearn.utils import class_weight

y_train_array = np.array(y_train)   # y_train 从Pandas库的数据格式（Series）转换成了NumPy库的数据格式（array）
# 为什么这么做？ 这是纯粹的技术兼容性问题。我们接下来要用的compute_class_weight这个工具，是Scikit-learn这个“工具箱”里的。
# 而这个工具箱里的工具，更习惯于处理NumPy格式的“零件”。所以，我们只是做了一个简单的格式转换，内容完全没变，就像把一份Word文档另存为PDF以方便另一个软件读取一样。

weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_array), y=y_train_array)  # 计算类别权重
# 调用一个功能强大的计算器。
# class_weight.compute_class_weight(...): 这是计算器的名字。
# 'balanced'意思是：“请你自动分析给你的数据，找出哪个类别是少数，哪个是多数，然后自动计算出一套权重，给少数派更高的权重，给多数派更低的权重。”
# classes=np.unique(y_train_array): 这是告诉计算器，我们的数据里有哪些类别。np.unique(y_train_array) 会返回 [0, 1]
# y=y_train_array: 这是把我们刚刚准备好的、纯数字格式的“标准答案”列表交给计算器，让它去数一数每个班级到底有多少人

print("类别权重:", weights)
class_weights = dict(enumerate(weights))# 将计算结果打包成 Keras 能够直接使用的格式。它不认[0.55, 5.0]，而是要求{0: 0.55, 1: 5.0}
print(f"为模型设定的类别权重: {class_weights}")

# --- END VERSION2 ---

# 开始训练！
# epochs=3 意味着模型会把全部训练数据完整地看3遍，每一遍都会优化自己
# validation_data 用来在训练过程中监控模型在测试集上的表现
history = model.fit(X_train_padded, y_train,
                    epochs=3,
                    validation_data=(X_test_padded, y_test),
                    class_weight=class_weights,  # <-- VERSION2：加入了赏罚机制
                    verbose=2)
print("模型训练完成。")



print("\n--- 步骤6：评估模型性能 ---")
# 这里的评估稍微不同。我们需要先把概率（0-1之间的数）转换成明确的0或1
threshold = 0.5
print(f"使用新的决策阈值: {threshold}")
y_pred_proba = model.predict(X_test_padded)  # 预测概率值，返回0-1之间的浮点数
y_pred = (y_pred_proba > threshold).astype(int)  # 使用0.5阈值转换为0/1二分类标签

report = classification_report(y_test, y_pred)
print("新模型评估报告:")
print(report)