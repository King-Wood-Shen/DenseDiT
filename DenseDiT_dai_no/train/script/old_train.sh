# Specify the config file path
export XFL_CONFIG=/home/users/astar/ares/qianhang/scratch/chengyou/sx/DenseDiT/DenseDiT_dai_no/train/config/config_densedit_stage2.yaml

# Specify the WANDB API key
export WANDB_API_KEY='5cad0341a1eb4333063066986b57550edb216815'

echo "Using config: $XFL_CONFIG"
export TOKENIZERS_PARALLELISM=true

# 设置DeepSpeed环境变量
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0

# 使用torchrun启动4卡分布式训练，使用DeepSpeed ZeRO Stage 2
torchrun --nproc_per_node=4 --master_port=29550 --nnodes=1 --node_rank=0 -m src.train.train