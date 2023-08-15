# LearnDeepSpeed 🚀
目的：基于DeepSpeed，突破硬件限制，实现大模型高效训练。

- 收录到[PytorchNetHub](https://github.com/bobo0810/PytorchNetHub)



# 最小示例

- [cifar示例](training/cifar/README.md)
  - 基于DeepSpeed的训练
  - MoE用法
- [pipeline_parallelism示例](training/pipeline_parallelism)
  - 流水并行的训练pipeline
  - 流水模型的保存、加载、指标评估
  - TensorBoard可视化



## 配置文件
- 等效batch计算
  ![img.png](assets/img.png)
- TensorBoard可视化
  ```json
  "tensorboard": {
    "enabled": true,  //开启可视化
    "output_path": "log/", //可视化文件保存路径
    "job_name": "2023年08月15日16:28:06" //此次实验名称，作为子文件夹
  }
  ```
  参考 [Link](https://www.deepspeed.ai/docs/config-json/#monitoring-module-tensorboard-wandb-csv)




# 参考

- [DeepSpeed官方Git库](https://github.com/microsoft/DeepSpeed)  

- [DeepSpeed官方文档](https://www.deepspeed.ai/getting-started/) 

- [DeepSpeed官方示例库](https://github.com/microsoft/DeepSpeedExamples)  

- [DeepSpeed基础用法](https://github.com/microsoft/DeepSpeedExamples/blob/master/training/HelloDeepSpeed/README.md) 