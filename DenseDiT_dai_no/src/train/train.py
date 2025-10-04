import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, DistributedSampler
import torch
import yaml
import os
import time
import json
import torch.distributed as dist
from .data import DenseDiTDataset
from .model import DenseDiTModel
from .callbacks import TrainingCallback

def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank

def get_world_size():
    try:
        world_size = int(os.environ.get("WORLD_SIZE"))
    except:
        world_size = 1
    return world_size

def get_config():
    config_path = os.environ.get("XFL_CONFIG")
    assert config_path is not None, "Please set the XFL_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def init_wandb(wandb_config, run_name):
    import wandb
    try:
        assert os.environ.get("WANDB_API_KEY") is not None
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config={},
        )
    except Exception as e:
        print("Failed to initialize WanDB:", e)

def main():
    # 初始化
    try:
        # 方法1: 检查DeepSpeedStrategy是否可用
        from pytorch_lightning.strategies import DeepSpeedStrategy
        deepspeed_available = True
        print("DeepSpeedStrategy is available")
    except ImportError:
        deepspeed_available = False
        print("DeepSpeedStrategy is not available")
    
    # 方法2: 直接检查deepspeed包
    try:
        import deepspeed
        print(f"DeepSpeed version: {deepspeed.__version__}")
        deepspeed_available = True
    except ImportError:
        print("DeepSpeed package is not installed")
        deepspeed_available = False
    rank = get_rank()
    world_size = get_world_size()
    is_main_process = (rank == 0)
    
    # torch.cuda.set_device(rank)
    config = get_config()
    training_config = config["train"]
    run_name = time.strftime("%Y%m%d-%H%M%S")

    # 初始化WanDB
    wandb_config = training_config.get("wandb", None)
    if wandb_config is not None and is_main_process:
        init_wandb(wandb_config, run_name)

    print(f"Rank: {rank}, World Size: {world_size}")
    if is_main_process:
        print("Config:", config)

    # 初始化数据集和dataloader
    def load_descriptions(description_file):
        descriptions = {}
        with open(description_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if " " not in line:
                    continue
                file_name, description = line.split(" ", 1)
                descriptions[file_name] = description
        return descriptions
    
    descriptions = load_descriptions("/home/users/astar/ares/qianhang/scratch/chengyou/sx/sx_data/rectify/image_descriptions.txt")
    
    dataset = DenseDiTDataset(
        image_dir="/home/users/astar/ares/qianhang/scratch/chengyou/sx/sx_data/rectify/pairs",
        condition_dir="/home/users/astar/ares/qianhang/scratch/chengyou/sx/sx_data/rectify/pairs_pf",
        context_file="/home/users/astar/ares/qianhang/scratch/chengyou/sx/sx_data/rectify/pairs",
        descriptions=descriptions,
    )

    print("Dataset length:", len(dataset))
    
    # 创建分布式采样器
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        sampler=sampler,
        shuffle=False,
        num_workers=training_config["dataloader_workers"],
        pin_memory=True,
        drop_last=True,
    )

    # 初始化模型
    trainable_model = DenseDiTModel(
        flux_pipe_id=config["flux_path"],
        device="cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
    )

    # Callbacks for logging and saving checkpoints
    training_callbacks = (
        [TrainingCallback(run_name, training_config=training_config)]
        if is_main_process
        else []
    )
    
    # # DeepSpeed ZeRO Stage 2 配置 - 这次确保使用
    # deepspeed_config = {
    #     "zero_optimization": {
    #         "stage": 2,
    #         "allgather_partitions": True,
    #         "allgather_bucket_size": 2e8,  # 减小bucket大小
    #         "reduce_scatter": True,
    #         "reduce_bucket_size": 2e8,
    #         "overlap_comm": True,
    #         "contiguous_gradients": True,
    #         "cpu_offload": False,
    #     },
    #     "fp16": {
    #         "enabled": config["dtype"] == "float16",
    #     },
    #     "bf16": {
    #         "enabled": config["dtype"] == "bfloat16",
    #     },
    #     "optimizer": {
    #         "type": "AdamW",
    #         "params": {
    #             "lr": training_config["optimizer"]["lr"],
    #             "betas": [0.9, 0.999],
    #             "eps": 1e-8,
    #             "weight_decay": 0.01,
    #         }
    #     },
    #     "gradient_accumulation_steps": training_config["accumulate_grad_batches"],
    #     "gradient_clipping": training_config.get("gradient_clip_val", 1.0),
    #     "train_batch_size": training_config["batch_size"] * world_size * training_config["accumulate_grad_batches"],
    #     "train_micro_batch_size_per_gpu": training_config["batch_size"],
    #     "wall_clock_breakdown": True,  # 改为True以显示更多信息
    #     "steps_per_print": 10,  # 更频繁的打印
    # }
    # 完整的ZeRO Stage 3配置（根据你的需求选择）
    deepspeed_config = {
        "zero_optimization": {
            "stage": 3,
            # 选项1: 完全在GPU上（最快，但内存占用最高）
            "offload_optimizer": {
                "device": "none"  # 保持在GPU上
            },
            "offload_param": {
                "device": "none"  # 保持在GPU上
            },
            
            # 选项2: 优化器offload到CPU（平衡性能和内存）
            # "offload_optimizer": {
            #     "device": "cpu",
            #     "pin_memory": True
            # },
            # "offload_param": {
            #     "device": "none"
            # },
            
            # 选项3: 全部offload到CPU（最省内存，但速度较慢）
            # "offload_optimizer": {
            #     "device": "cpu", 
            #     "pin_memory": True
            # },
            # "offload_param": {
            #     "device": "cpu",
            #     "pin_memory": True
            # },
            
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        },
        "fp16": {
            "enabled": config["dtype"] == "float16",
            "auto_cast": False,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "bf16": {
            "enabled": config["dtype"] == "bfloat16"
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": training_config["optimizer"]["lr"],
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": training_config["optimizer"].get("weight_decay", 0.01),
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": training_config["optimizer"]["lr"],
                "warmup_num_steps": training_config.get("warmup_steps", 1000),
            }
        },
        "gradient_accumulation_steps": training_config["accumulate_grad_batches"],
        "gradient_clipping": training_config.get("gradient_clip_val", 1.0),
        "train_batch_size": training_config["batch_size"] * world_size * training_config["accumulate_grad_batches"],
        "train_micro_batch_size_per_gpu": training_config["batch_size"],
        "wall_clock_breakdown": True,
        "steps_per_print": 10,
        "dump_state": False,
    }
    
    # 初始化trainer with DeepSpeed ZeRO Stage 2
    strategy = DeepSpeedStrategy(
        config=deepspeed_config,  # 这里正确传递配置
        logging_batch_size_per_gpu=training_config["batch_size"],
        allgather_bucket_size=deepspeed_config["zero_optimization"]["allgather_bucket_size"],
        reduce_bucket_size=deepspeed_config["zero_optimization"]["reduce_bucket_size"],
    )
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=world_size,
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=training_callbacks,
        enable_checkpointing=True,
        enable_progress_bar=is_main_process,  # 只在主进程显示进度条
        logger=True,
        strategy=strategy,  # 使用DeepSpeed策略
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
        precision="bf16" if config["dtype"] == "bfloat16" else "16",
    )

    setattr(trainer, "training_config", training_config)

    # 保存配置
    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}", exist_ok=True)
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)
        
        # 保存DeepSpeed配置
        with open(f"{save_path}/{run_name}/deepspeed_config.json", "w") as f:
            json.dump(deepspeed_config, f, indent=2)

    # 开始训练
    print(f"Starting training with DeepSpeed ZeRO Stage 2...")
    trainer.fit(trainable_model, train_loader)

if __name__ == "__main__":
    main()