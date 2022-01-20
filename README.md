# StyleGanJittor (Tsinghua university computer graphics course)
Jittor Implementation of StyleGAN (Tsinghua university computer graphics course)
这个项目是我用计图（jittor）实现的 StyleGAN。StyleGAN是 NVIDIA 在 2018 年提出的一种用于图像生成的生成对抗网络。该网络模型相对于此前的模型的主要改进点在于生成器 (generator) 的结构，包括加入了一个八层的 Mapping Network，使用了 AdaIn 模块以及引入了图像随机性——这些结构使得生成器可以将图像的整体特征与局部特征进行解耦，从而合成效果更好的图像；同时网络也具有更优的隐空间插值效果。
