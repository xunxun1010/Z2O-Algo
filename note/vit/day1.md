## vit用途
<pre>
import torch
from vit_pytorch import ViT

v = ViT(
    image_size = 256,       #输入图像的尺寸。
    patch_size = 32,        #图像块（Patch）的大小。
    num_classes = 1000,     #模型需要分类的类别总数。
    dim = 1024,             #线性变换后的输出张量维度（即Transformer内部的隐藏层特征维度）。
    depth = 6,              #Transformer 编码器块（Blocks）的层数。
    heads = 16,             #多头注意力（Multi-head Attention）机制中的“头”数。
    mlp_dim = 2048,         #Transformer块内部的MLP（多层感知机/前馈神经网络）的维度。
    dropout = 0.1,          #Dropout丢弃率
    emb_dropout = 0.1       #嵌入层（Embedding）的丢弃率，防止模型过拟合。
)

img = torch.randn(1, 3, 256, 256)   #这里生成了一个形状为 (批次大小, 通道数, 高度, 宽度) 的随机张量，用来模拟一张输入图片 。1代表这张图片是一个批次（Batch size = 1），3代表RGB 三个颜色通道,256, 256 对应前面设置的image_size。

preds = v(img) # (1, 1000)          #将模拟图片张量输入到构建好的 ViT 模型中进行前向传播（Forward pass）。最终输出的 preds 形状为 (1, 1000)，代表模型对这 1 张图片在 1000 个类别上的预测概率得分
</pre>
