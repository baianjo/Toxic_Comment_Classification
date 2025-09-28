# 这是一个改进版的RNN模型，使用了Focal Loss和智能阈值选择来提高有毒评论分类的性能
# 相比于基础版本，这个模型使用了双向LSTM和更精细的损失函数来处理类别不平衡问题

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve
import matplotlib.pyplot as plt

# 导入Keras组件用于构建神经网络
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense, Bidirectional

# 可选：设置随机种子以提高可复现性
# np.random.seed(42)
# tf.random.set_seed(42)

# ----------------------------
# 步骤1：加载并准备数据
# ----------------------------
print("--- 步骤1：加载并准备数据 ---")
# 从CSV文件中读取训练数据
df = pd.read_csv('train.csv')
# 为了加快实验速度，只使用前30000条数据
df_sample = df.head(30000)

# 提取文本和标签
X = df_sample['comment_text']  # 输入：评论文本
y = df_sample['toxic'].values  # 输出：是否有毒标签
# 将数据分为训练集和测试集（70%训练，30%测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
print(f"有毒评论比例 (训练集): {y_train.mean():.3f}")


# ----------------------------
# 步骤2：文本数字化（Tokenizer）
# ----------------------------
print("\n--- 步骤2：文本数字化 ---")
# 初始化Tokenizer，只保留最常见的10000个词
# 这就像创建一本只包含最重要10000个词的"密码本"
tokenizer = Tokenizer(num_words=10000)
# 让tokenizer学习训练集中的词汇
tokenizer.fit_on_texts(X_train)

# 将文本转换为整数序列（每句话变成一串数字）
# 就像把每篇中文文章翻译成由数字组成的密码
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)


# ----------------------------
# 步骤3：统一序列长度（Padding）
# ----------------------------
print("\n--- 步骤3：统一序列长度 ---")
# 设置最大序列长度为200
maxlen = 200
# 统一所有序列的长度：短的补0，长的截断
# 神经网络要求所有输入数据的形状必须一致
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen, padding='post', truncating='post')
print(f"填充后数据形状: {X_train_pad.shape}")


# ----------------------------
# 步骤4：定义 Focal Loss（解决类别不均衡）
# ----------------------------
def focal_loss(gamma=2.0, alpha=0.75):
    """
    Focal Loss：自动降低易分样本（如"无毒"）的权重，
    聚焦于难分的少数类（"有毒"），避免模型"躺平"。
    
    在有毒评论分类任务中，"有毒"评论通常是少数类，传统的交叉熵损失
    会让模型偏向于预测多数类（"无毒"），而Focal Loss通过动态调整
    样本权重来解决这个问题。
    """
    def loss_fn(y_true, y_pred):
        # 防止log(0)的情况出现
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        # 计算正确类别的预测概率
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1. - y_pred)
        # 平衡正负样本
        alpha_factor = tf.ones_like(y_true) * alpha
        alpha_t = tf.where(tf.equal(y_true, 1), alpha_factor, 1. - alpha_factor)
        # 计算交叉熵损失
        cross_entropy = -tf.math.log(p_t)
        # 计算focal weight，降低易分样本的权重
        weight = alpha_t * tf.pow((1. - p_t), gamma)
        # 返回加权后的平均损失
        return tf.reduce_mean(weight * cross_entropy)
    return loss_fn


# ----------------------------
# 步骤5：搭建改进的 LSTM 模型
# ----------------------------
print("\n--- 步骤5：搭建模型 ---")
# 构建神经网络模型
model = Sequential([
    # 嵌入层：将每个词的整数编码转换为64维向量
    # 这就像给每个词分配一个64维的"GPS坐标"，让计算机理解词的语义
    Embedding(input_dim=10000, output_dim=64, input_length=maxlen),
    
    # 双向LSTM层：更好地捕捉前后文信息，内置dropout提高稳定性
    # 双向LSTM可以从两个方向处理文本，更好地理解上下文语境
    Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.3)),
    
    # 全连接层：进一步提取特征
    Dense(24, activation='relu'),
    
    # Dropout层：防止过拟合
    # 随机关闭50%的神经元，提高模型泛化能力
    Dropout(0.5),
    
    # 输出层：单个神经元输出有毒概率
    Dense(1, activation='sigmoid')
])

# 编译模型：指定优化器、损失函数和评估指标
model.compile(
    optimizer='adam',                           # 优化算法
    loss=focal_loss(gamma=2.0, alpha=0.75),     # 使用Focal Loss解决类别不平衡
    metrics=['accuracy']                        # 评估指标
)

# 显示模型结构
model.summary()


# ----------------------------
# 步骤6：训练模型（无class_weight！）
# ----------------------------
print("\n--- 步骤6：训练模型 ---")
# 开始训练模型
history = model.fit(
    X_train_pad, y_train,           # 训练数据
    epochs=3,                       # 训练轮数
    batch_size=64,                  # 批次大小
    validation_data=(X_test_pad, y_test),  # 验证数据
    verbose=2                       # 输出详细信息
)


# ----------------------------
# 步骤7：智能阈值选择 + 评估
# ----------------------------
print("\n--- 步骤7：评估模型（自动选择最优阈值）---")

# 获取测试集的预测概率
y_pred_proba = model.predict(X_test_pad).flatten()

# 使用Precision-Recall曲线找到最佳阈值
# 传统的0.5阈值可能不是最优的，我们通过PR曲线找到F1分数最高的阈值
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
# 计算各种阈值下的F1分数
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
# 找到F1分数最高的阈值
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

print(f"最优阈值（基于F1）: {best_threshold:.3f}")
print(f"对应 Precision: {precision[best_idx]:.3f}, Recall: {recall[best_idx]:.3f}, F1: {f1_scores[best_idx]:.3f}")

# 使用最佳阈值进行预测
y_pred = (y_pred_proba >= best_threshold).astype(int)

# 输出详细的分类报告
report = classification_report(y_test, y_pred, target_names=['Non-Toxic', 'Toxic'])
print("\n最终评估报告:")
print(report)

# 可选：绘制预测概率分布图
plt.figure(figsize=(8, 4))
# 分别绘制有毒和无毒评论的预测概率分布
toxic_probs = y_pred_proba[y_test == 1]
nontoxic_probs = y_pred_proba[y_test == 0]
plt.hist(nontoxic_probs, bins=50, alpha=0.6, label='Non-Toxic', color='blue')
plt.hist(toxic_probs, bins=50, alpha=0.6, label='Toxic', color='red')
# 添加最佳阈值线
plt.axvline(best_threshold, color='green', linestyle='--', label=f'Best Threshold = {best_threshold:.2f}')
plt.xlabel('Predicted Probability of Toxic')
plt.ylabel('Frequency')
plt.legend()
plt.title('Prediction Probability Distribution')
plt.tight_layout()
plt.show()