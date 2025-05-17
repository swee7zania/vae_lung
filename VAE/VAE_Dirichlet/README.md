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

**数据加载模块 (`data_loader.py`):**

- 读取 `latent_vectors_{Run}.npy` 文件。
- 根据 `meta_mal_nonmal.csv` 划分数据集（支持交叉验证）。

**模型模块 (`mlp_model.py`):**

- 动态生成 MLP 网络结构，根据超参数 `Depth` 决定网络深度。

**训练模块 (`trainer.py`):**

- 包含 `train()` 和 `evaluate()` 方法，用于交叉验证训练和评估。

**配置模块 (`config.py`):**

- 包含 VAE 和 MLP 模型的超参数空间定义，以及 `get_random_hyperparams()` 方法，用于随机采样超参数。

**主程序 (`main.py`):**

- 加载潜在向量，进行数据分割。
- 根据采样的超参数构建并训练 MLP 模型。