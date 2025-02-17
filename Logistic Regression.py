import numpy as np

# 定义 sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 定义损失函数
def compute_loss(y, t):
    return -np.mean(t * np.log(y) + (1 - t) * np.log(1 - y))

# 初始化权重
w = np.array([-0.6, 2, 1.2, -2.8])

# 学习率
learning_rate = 0.1

# 输入数据
X = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0]
])

# 目标标签
t = np.array([0, 1, 1])

# 迭代次数
num_iterations = 1000

# 记录最后 10 次更新的结果
last_updates = []

for i in range(num_iterations):
    # 计算 z 和 y
    z = np.dot(X, w[1:]) + w[0]
    y = sigmoid(z)
    
    # 计算损失
    loss = compute_loss(y, t)
    
    # 计算梯度
    dw0 = np.mean(y - t)
    dw1 = np.mean((y - t) * X[:, 0])
    dw2 = np.mean((y - t) * X[:, 1])
    dw3 = np.mean((y - t) * X[:, 2])
    
    # 更新权重
    w[0] -= learning_rate * dw0
    w[1] -= learning_rate * dw1
    w[2] -= learning_rate * dw2
    w[3] -= learning_rate * dw3
    
    # 记录最后 10 次更新的结果
    if i >= num_iterations - 10:
        last_updates.append((z, y, loss))

# 输出最后 10 次更新的结果
for update in last_updates:
    print(f"z: {update[0]}, y: {update[1]}, Loss: {update[2]}")