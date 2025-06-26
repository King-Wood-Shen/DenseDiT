import os
from tqdm import tqdm
import torch
from diffusers.pipelines import FluxPipeline
from src.flux.condition import Condition
from PIL import Image

from src.flux.generate import generate, seed_everything

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")

pipe.load_lora_weights(
    "ptah/to/the/lora/weights/of/the/denseworld/task",
    adapter_name="densedit",
)

# Set the height and width for the generated images of the denseworld task
height = 512
width = 512
# Set the path to save the results
savepath = 'path/to/save/the/inference/results/of/the/denseworld/task'

for root, dirs, files in os.walk('path/to/the/test/images/of/the/denseworld/task'):
    for name in tqdm(files, desc="Processing files", unit="file"):
        file_path = os.path.join(root, name)
        image = Image.open(file_path).convert("RGB")
        prompt = "The prompt for the task"
        condition = Condition("densedit", condition = image)
        seed_everything()
        result_img = generate(
            pipe,
            prompt=prompt,
            conditions=[condition],
            height = height,
            width = width
        ).images[0]
        result_img.save(os.path.join(savepath, name.replace('jpg', 'png')))