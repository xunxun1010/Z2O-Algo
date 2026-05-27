### 《Decouple-to-fuse: Saliency guided dual-branch framework for underwater image enhancement and super resolution》
source：https://www.sciencedirect.com/science/article/pii/S1566253526003283  

### 1.论文的核心研究问题
致力于解决水下图像增强与超分辨率（UIESR）联合任务中的“耦合冲突”与“空间异质性”问题。  
现有方法通常在一个共享特征空间内同时处理颜色校正（低频平滑）和纹理重建（高频锐化），导致两者相互制约（负迁移）；且现有方法多采用全局统一策略，无法适应水下场景中前景目标与浑浊背景退化程度不同的空间异质性特点。作者旨在通过任务解耦和显著性引导，实现既色彩自然又纹理清晰的水下图像复原。
