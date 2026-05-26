git远程clone，代码初步测试

1.Paired Translation with pix2pix-turbo

2.Unpaired Translation with CycleGAN-Turbo

3.transformer论文阅读  
核心思想：完全基于注意力机制，摒弃循环和卷积  
在 Transformer 之前，主流的序列模型（如 RNN、LSTM、GRU）都依赖循环结构来处理序列数据，而 CNN 虽然可以并行，但在捕捉长距离依赖方面效果有限。这些模型存在两个主要问题：  
1. 难以并行化（RNN 必须按顺序计算）  
2. 长距离依赖建模困难  
Transformer 的核心创新在于：完全抛弃了循环和卷积结构，仅使用“自注意力机制”（Self-Attention）来建模序列中任意两个位置之间的关系。  
模型架构概览  
Transformer 采用经典的 Encoder-Decoder 结构，但每一层都由以下关键组件构成：  

1. 自注意力机制（Self-Attention）  
● 让序列中的每个词都能“关注”到其他所有词。  
● 通过计算 Query（查询）、Key（键）、Value（值） 三者之间的相关性，动态加权聚合信息。  
● 公式核心  
   
2.多头注意力（Multi-Head Attention）
● 将注意力机制“并行”多次，从不同子空间学习不同的表示，再拼接起来。  
● 增强模型捕捉不同位置、不同语义关系的能力。    
3. 位置编码（Positional Encoding）  
● 因为没有循环结构，模型不知道词的顺序。  
● 所以通过正弦/余弦函数或可学习的向量，将位置信息注入输入嵌入中。  
4. 前馈神经网络（Position-wise Feed-Forward Network）  
● 每个位置独立经过一个两层全连接网络，增加非线性表达能力。  
5. 残差连接 + 层归一化（Residual Connection & LayerNorm）  
● 每个子层后都加残差连接和 LayerNorm，帮助训练深层网络。  
主要优势
● 高度并行化：不再受制于序列顺序，训练速度大幅提升。  
● 长距离依赖建模能力强：任意两个词之间直接建立联系。  
● 性能优越：在机器翻译任务（如 WMT 2014 English-to-German / English-to-French）上达到当时 SOTA，且训练时间远少于 RNN/CNN 模型。  
