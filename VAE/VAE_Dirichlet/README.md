# 分类实现

`RandomSearch_Dirichlet_VAE.py` 中的分类是通过 **VAE 模型生成的潜在向量 (latent vectors)**，基于这些潜在向量构建 **MLP (Multi-Layer Perceptron) 分类器** 实现的。主要有以下几个步骤：

1. VAE 模型训练并生成潜在向量
2. 数据准备：生成分类数据集
3. MLP 分类器架构定义
4. 交叉验证：数据集划分与训练
5. 测试与评估



```py
/VAE/
├── dirichlet_vae.py  # VAE模型类
├── trainer.py        # VAE训练类
├── data_loader.py    # 数据加载模块
├── config.py         # 超参数配置模块
├── main.py           # 主程序入口

/MLP/
├── mlp_model.py      # MLP模型类
├── trainer.py        # MLP训练类
├── data_loader.py    # 数据加载模块
├── config.py         # 超参数配置模块
├── main.py           # 主程序入口
```

### 🔶 **1. 特征来源：VAE的Dirichlet分布输出**

- 使用训练好的变分自编码器（`DIR_VAE`）对图像进行编码，提取 **Dirichlet参数（`alpha`）作为潜在向量**。
- 将这些潜在向量作为MLP的输入特征（见 `latent_vectors`）。

------

### 🔶 **2. 标签来源与数据准备**

- 标签文件（如 `labels2.npy`, `labels3.npy`）提供分类任务的标签（如恶性/非恶性或恶性/良性）。
- 相关元数据来自CSV文件（如 `meta_mal_nonmal.csv`），用于患者分组做交叉验证。
- 在函数 `data_split()` 中，患者ID被划分到不同fold里，确保每个fold是**病人独立的**。

------

### 🔶 **3. 模型结构定义**

定义于函数 `train_model()` 中：

- **MLP使用`nn.Sequential`定义**，支持4层或5层网络（根据超参数“Depth”）：

  ```python
  model = nn.Sequential(
      nn.Linear(input_dim, hidden1),
      nn.GELU(),
      nn.BatchNorm1d(...),
      nn.Dropout(...),
      ...
      nn.Linear(..., 1),
      nn.Sigmoid()
  )
  ```

- 输入维度为 `latent_size * base`，即来自VAE的潜在表示长度。

------

### 🔶 **4. 训练细节**

- 损失函数：**BCELoss**（二分类交叉熵）。
- 优化器：**Adam**，支持 `ReduceLROnPlateau` 学习率调度。
- 支持 `early_stopping()` 防止过拟合。
- 每次训练记录：训练损失、训练准确率、验证损失、验证准确率。

------

### 🔶 **5. 模型评估**

- 使用函数 `stats()` 和 `confusion_matrix()` 评估模型在验证集或测试集上的表现。
- 评估指标包括：
  - Accuracy
  - Precision
  - Recall
  - Specificity
  - F1 Score
  - AUC（Area Under Curve）

------

### 🔶 **6. 超参数搜索**

- 使用函数 `test_hyperparams()` 对MLP超参数进行随机搜索（如层大小、Dropout、学习率、阈值等）。
- 每组超参数都通过5折交叉验证进行评估。
- 根据AUC或Accuracy选择最优模型。