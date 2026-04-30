Simple ViT

```python
import torch
from vit_pytorch import SimpleViT

v = SimpleViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,          # Transformer 内部的隐藏层特征维度
    depth = 6,           # Transformer 块的层数
    heads = 16,          # 多头注意力机制的“头”数
    mlp_dim = 2048       # Transformer 块内部的前馈神经网络维度
)

img = torch.randn(1, 3, 256, 256)

preds = v(img) # (1, 1000)
```
## 一、note

### 1.架构上  
1.弃用CLS Token，改用全局平均池化 (Global Average Pooling)：原版 ViT 在输入序列的开头硬塞了一个特殊的 [CLS] 标签，专门用来收集全局信息做分类。Simple ViT 觉得这很麻烦，直接在最后一层把所有图像块（Patch）的特征求个平均值，用这个平均特征来做分类。  
2.改用2D正弦位置编码 (2D Sinusoidal Positional Embedding)：原版用的是可学习的 1D 位置编码。Simple ViT 改成了基于正弦/余弦函数的 2D 固定编码，这更符合图片长宽二维的物理直觉。  
3.彻底放弃Dropout(No dropout)：原版为了防止过拟合加了 Dropout，Simple ViT 直接把它全删了，网络变得更干净。  
4.极简分类头(Simple Linear Head)：原版在网络最后用了一个多层感知机（MLP）来输出类别，这里证明了直接用一个单层的线性分类器（Linear layer）效果也毫不逊色。

### 2.训练策略  
1.批次大小：原版训练需要一次性塞入 4096 张图片（Batch size = 4096），极其吃显卡算力。Simple ViT证明了把 Batch size降到1024依然可行，这大大降低了硬件门槛。  
2.数据增强：配合使用了 RandAugment（随机图像变换组合）和 MixUp（把两张图片和它们的标签按比例混合）等数据增强技术，逼迫模型学到更本质的特征。  
## 二、test
```python
# 1. 查看张量的形状
print(" 1. 结果的形状")
print(preds.shape)

# 2. 查看原始的输出数值（Logits）
print("\n 2. 原始得分 (前10个) ")
print(preds[0, :10]) 

# 3. 找到模型预测的最终类别
predicted_class_index = torch.argmax(preds, dim=-1)
print("\n 3. 最终预测结果 ")
print("预测的类别索引:", predicted_class_index.item())

# 4. 将得分转换为概率百分比
probabilities = torch.softmax(preds, dim=-1)
max_prob = torch.max(probabilities).item()
print(f"该类别的置信度 (概率): {max_prob * 100:.2f}%")

```
### notes
每次预测的类别都在变，且最高概率往往非常低（比如只有 0.1% 左右）。这是因为你实例化的 v 是一个未经任何训练、参数完全随机初始化的“婴儿模型”，而你输入的 img 也是用 torch.randn 生成的随机雪花噪点图。

这段代码的核心意义是帮你跑通前向传播（Forward Pass）的数据流转过程。在实际工程中，我们需要给模型加载训练好的“预训练权重（Pre-trained weights）”，并传入用 cv2 或 PIL 读取的真实猫狗图片，那个时候输出的测试结果才具备真正的现实意义。




