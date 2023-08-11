#!/bin/bash

# Number of nodes 节点数
NUM_NODES=1
# Number of GPUs per node  每个节点的GPU数
NUM_GPUS=2
# Size of expert parallel world (should be less than total world size)  专家并行度
EP_SIZE=2
# Number of total experts  列表，指明每层的专家数
EXPERTS=2

deepspeed --num_nodes=${NUM_NODES} --num_gpus=${NUM_GPUS} cifar10_deepspeed.py \
	--log-interval 100 \
	--deepspeed \
	--deepspeed_config ds_config.json \
	--moe \
	--ep-world-size ${EP_SIZE} \
	--num-experts ${EXPERTS} \
	--top-k 1 \
	--noisy-gate-policy 'RSample' \
	--moe-param-group