#!/bin/bash

deepspeed cifar10_deepspeed.py --deepspeed --deepspeed_config ds_config.json $@ 


# 1. 运行脚本并传递参数时，$@ 会捕获所有的参数，并可以在脚本中使用。
# 2. --deepspeed 是否启动DeepSpeed
# 3. --deepspeed_config ds_config.json 传递配置文件