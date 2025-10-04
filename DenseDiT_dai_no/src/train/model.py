import lightning as L
from diffusers.pipelines import FluxKontextPipeline  # 替换为 FluxKontextPipeline
import torch
from peft import LoraConfig, get_peft_model_state_dict

import prodigyopt

# from ..flux1.transformer import tranformer_forward
from ..flux.condition import Condition
from ..flux.pipeline_tools import encode_images, prepare_text_input


class DenseDiTModel(L.LightningModule):
    def __init__(
        self,
        flux_pipe_id: str,
        # lora_path: str = None,
        # lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        # Load the FluxKontext pipeline
        self.flux_pipe: FluxKontextPipeline = (
            FluxKontextPipeline.from_pretrained(flux_pipe_id).to(dtype=dtype).to(device)
        )
        self.transformer = self.flux_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
        self.transformer.train()
        self.save_hyperparameters(ignore=['flux_pipe_id'])
        # Freeze the FluxKontext pipeline components
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()
        
        # unfreeze the transformerblocks
        # self.unfreeze_transformer_blocks()
        
        # Initialize LoRA layers
        # self.lora_layers = self.init_lora(lora_path, lora_config)

        self.to(device).to(dtype)

    # def unfreeze_transformer_blocks(self):
    #     """Unfreeze only the transformer block layers."""
    #     for i, block in enumerate(self.transformer.blocks):
    #         if i >= 0: 
    #             for param in block.parameters():
    #                 param.requires_grad = True
    # def init_lora(self, lora_path: str, lora_config: dict):
    #     assert lora_path or lora_config
    #     if lora_path:
    #         from peft import set_peft_model_state_dict
    #         from safetensors.torch import load_file

    #         self.transformer.add_adapter(LoraConfig(**lora_config))
    #         state_dict = load_file(lora_path)
    #         set_peft_model_state_dict(self.transformer, state_dict)
    #         lora_layers = filter(lambda p: p.requires_grad, self.transformer.parameters())
    #     else:
    #         self.transformer.add_adapter(LoraConfig(**lora_config))
    #         # TODO: Check if this is correct (p.requires_grad)
    #         lora_layers = filter(
    #             lambda p: p.requires_grad, self.transformer.parameters()
    #         )
    #     return list(lora_layers)

    # def save_lora(self, path: str):
    #     FluxKontextPipeline.save_lora_weights(
    #         save_directory=path,
    #         transformer_lora_layers=get_peft_model_state_dict(self.transformer),
    #         safe_serialization=True,
    #     )

    def configure_optimizers(self):
        # Freeze the transformer
        # self.transformer.requires_grad_(False)
        # print("self.transformer", type(self.transformer), self.transformer)
        # for param in self.transformer_blocks.parameters():
        #     print(param.dtype)
        all_params = []
        for block in self.transformer.transformer_blocks:
            for param in block.parameters():
                print("111", param.dtype)
                all_params.append(param)

        # for block in self.transformer.single_transformer_blocks:
        #     for param in block.parameters():
        #         all_params.append(param)
        print("all_params", len(all_params))
        for p in all_params:
            p.requires_grad_(True)
        self.trainable_params = all_params
        opt_config = self.optimizer_config
        print("self", len(self.trainable_params))
        # Set the trainable parameters
        # self.trainable_params = self.lora_layers

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_idx):
        step_loss = self.step(batch)
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def step(self, batch):
        imgs = batch["image"]
        conditions = batch["condition"]
        context = batch["context"]
        prompts = batch["description"]

        # Prepare inputs
        with torch.no_grad():
            # Prepare image input
            x_0, img_ids = encode_images(self.flux_pipe, imgs)
                
            # Prepare text input    
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                self.flux_pipe, prompts
            )

            # Prepare t and x_t
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device)
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)

            # Prepare conditions
            condition_latents, condition_ids = encode_images(self.flux_pipe, conditions)
            # print("condition_ids", condition_ids)
            # Prepare context
            context_latents, context_ids = encode_images(self.flux_pipe, context)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )
            latent_ids = torch.cat([img_ids, condition_ids, context_ids], dim=0)  # dim 0 is sequence dimension
            latent_model_input = torch.cat([x_t, condition_latents, context_latents], dim=1)
        # Forward pass
        transformer_out = self.transformer(
            # Inputs to the original transformer
            hidden_states=latent_model_input,
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        pred = transformer_out[0]
        pred = pred[:, : x_t.size(1)]
        # Compute loss
        loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        self.last_t = t.mean().item()
        return loss
