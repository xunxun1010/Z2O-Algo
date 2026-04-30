NaViT
```python
import torch
from vit_pytorch.na_vit import NaViT

# 实例化 NaViT (Native Resolution ViT) 模型
v = NaViT(
    image_size = 256,         # 基础图像尺寸。虽说是原生多分辨率，但底层仍需要一个参考尺寸来计算基础参数
    patch_size = 32,          
    num_classes = 1000,       
    dim = 1024,               # Transformer 内部的隐藏层特征维度
    depth = 6,                # Transformer 编码器块（Blocks）的层数，代表网络的深度
    heads = 16,               # 多头注意力（Multi-head Attention）机制的“头”数
    mlp_dim = 2048,           # Transformer 块内部前馈神经网络（MLP）的维度
    dropout = 0.1,            # 常规的 Dropout 丢弃率，用于防止模型过拟合
    emb_dropout = 0.1,        # 嵌入层（Embedding）的丢弃率
    token_dropout_prob = 0.1  # NaViT 特有的机制：Token 丢弃率。这里设置为丢弃 10% 的 Token（保留 90%），能加速训练并充当正则化
)

# 模拟生成 5 张不同分辨率的图片，并采用序列打包 (Sequence Packing) 的特殊输入格式：List[List[Tensor]] 
# 注意：目前你需要手动将图片拼凑在同一个 Batch 元素子列表中。
# 拼凑的原则是：同一个子列表内的图片切块后，Token 总数不能超出模型计算自注意力时的最大允许序列长度
images = [
    # 第一个 Batch 元素：内部包含 2 张图片（一张 256x256，一张 128x128）  
    [torch.randn(3, 256, 256), torch.randn(3, 128, 128)],
    
    # 第二个 Batch 元素：内部包含 2 张图片（一张 128x256，一张 256x128）  
    [torch.randn(3, 128, 256), torch.randn(3, 256, 128)],
    
    # 第三个 Batch 元素：内部只包含 1 张图片（一张 64x256）  
    [torch.randn(3, 64, 256)]
]

# 进行前向传播
preds = v(images) 

# 输出结果张量维度为 (5, 1000)
# 解释：虽然外层列表只有 3 个元素，但模型内部通过掩码机制成功识别并处理了总共 5 张不同分辨率的图片，输出了这 5 张图片的 1000 类预测得分



```
