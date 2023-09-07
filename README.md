# LearnDeepSpeed 🚀
目的：基于DeepSpeed，突破硬件限制，实现大模型高效训练。

- 收录到[PytorchNetHub](https://github.com/bobo0810/PytorchNetHub)



# 最小示例

- [cifar示例](training/cifar/README.md)
  - 分布式数据并行DDP的训练pipeline
  - MoE用法
  - 学习率调度器的配置
  - ZeRO零冗余优化器的配置
- [pipeline_parallelism示例](training/pipeline_parallelism)
  - 流水并行的训练pipeline
  - 流水模型的保存、加载、指标评估
  - TensorBoard可视化




## DeepSpeed训练Tricks
https://zhuanlan.zhihu.com/p/654923210


## DeepSpeed训练配置

https://zhuanlan.zhihu.com/p/654925843






# 参考

- [官方Git库](https://github.com/microsoft/DeepSpeed)  
- [官方文档](https://www.deepspeed.ai/getting-started/) 
- [官方示例库](https://github.com/microsoft/DeepSpeedExamples)  
- [DeepSpeed入门教程](https://zhuanlan.zhihu.com/p/630734624)   