import lightning as L
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from transformers import pipeline
import cv2
import torch
import os
from diffusers.utils import load_image

try:
    import wandb
except ImportError:
    wandb = None

from ..flux.condition import Condition
from ..flux.generate import generate


class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )

        self.total_steps = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                "batch": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            wandb.log(report_dict)

        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Save LoRA weights at specified intervals
        if self.total_steps % self.save_interval == 0:
            # print(
            #     f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            # )
            # pl_module.save_lora(
            #     f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            # )
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving transformer weights"
            )
            trainer.save_checkpoint(f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}")

        try:
            # Generate and save a sample image at specified intervals
            # 只在主进程（rank 0）中执行采样，避免分布式训练中的重复采样
            if self.total_steps % self.sample_interval == 0 and trainer.is_global_zero:
                print(
                    f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
                )
                self.generate_a_sample(
                    trainer,
                    pl_module,
                    f"{self.save_path}/{self.run_name}/output",
                    f"sample_{self.total_steps}",
                )
        except Exception as e:
            print(f"Error generating sample: {e}")

    @torch.no_grad()
    def generate_a_sample(
        self,
        trainer,
        pl_module,
        save_path,
        file_name,
    ):
        # 确保只在主进程中执行
        if not trainer.is_global_zero:
            return
            
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 使用模型的设备，确保与训练设备一致
        device = next(pl_module.parameters()).device
        generator = torch.Generator(device=device).manual_seed(42)
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        descriptions = []
        with open('/home/users/astar/ares/qianhang/scratch/chengyou/sx/sx_data/rectify/image_descriptions.txt', 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    columns = line.split(" ", 1)
                    if len(columns) >= 2:
                        first_col = columns[0]
                        second_col = columns[1]
                        descriptions.append((first_col, second_col))
        
        # 随机选择5个例子（减少内存使用）
        import random
        random.seed(42)  # 设置随机种子确保可重复性
        selected_descriptions = random.sample(descriptions, min(5, len(descriptions)))
        print(f"从{len(descriptions)}个例子中随机选择了{len(selected_descriptions)}个进行采样")
        print("generator:", generator, generator.device)
        print("model_config:", pl_module.model_config)
        
        # 遍历选中的5个例子进行采样
        for i, (file_name, description) in enumerate(selected_descriptions):
            try:
                # 构建图片路径
                
                context_path = f"/home/users/astar/ares/qianhang/scratch/chengyou/sx/sx_data/rectify/pairs/{file_name}.jpg"
                tail = first_col.split('_')[-1]
                if tail == 'right':
                    first_col = first_col.replace('_right', '_left')
                else:
                    first_col = first_col.replace('_left', '_right')                
                condition_path = f"/home/users/astar/ares/qianhang/scratch/chengyou/sx/sx_data/rectify/pairs_pf/{first_col}_pf.jpg"
                # 检查文件是否存在
                if not os.path.exists(condition_path) or not os.path.exists(context_path):
                    print(f"跳过 {file_name}: 图片文件不存在")
                    continue

                # 加载两张输入图片
                condition_img = load_image(condition_path)
                context_img = load_image(context_path)

                # 创建 Condition 对象，包含两张图片
                condition = Condition(
                    condition=condition_img,
                    context=context_img,
                )
                
                # 使用 generate 函数生成图像
                res = generate(
                    pl_module.flux_pipe,
                    prompt=description,
                    conditions=[condition],
                    height=1024,
                    width=1024,
                    guidance_scale=3.5,
                    generator=generator,
                    model_config=pl_module.model_config,
                    default_lora=True,
                )

                # 保存生成的图像
                out_path = os.path.join(save_path, f"{first_col}.png")
                res.images[0].save(out_path)
                print(f"保存样本 {i+1}/{len(selected_descriptions)}: {out_path}")
                
                # 清理内存
                del res
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"处理 {file_name} 时出错: {e}")
                continue
