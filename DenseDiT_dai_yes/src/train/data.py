from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
    
class DenseDiTDataset(Dataset):
    def __init__(
        self,
        image_dir,
        condition_dir,
        descriptions,
        condition_size: int = 512,
        target_size: int = 512,
        condition_type: str = "densedit",
        drop_text_prob: float = 0,
        drop_image_prob: float = 0,
        return_pil_image: bool = False,
    ):
        self.image_dir = image_dir
        self.condition_dir = condition_dir
        self.descriptions = descriptions
        self.file_names = list(descriptions.keys())
        self.condition_size = condition_size
        self.target_size = target_size
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image

        self.to_tensor = T.ToTensor()
    
    def load_images(self, image_dir, condition_dir, file_name):
        image_path = os.path.join(image_dir, f"{file_name}.png")
        condition_path = os.path.join(condition_dir, f"{file_name}.jpg")

        image = Image.open(image_path).convert("RGB")
        condition_image = Image.open(condition_path).convert("RGB")

        return image, condition_image

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        description = self.descriptions[file_name]

        image, condition_img = self.load_images(self.image_dir, self.condition_dir, file_name)

        return {
            "image": self.to_tensor(image),
            "condition": self.to_tensor(condition_img),
            "condition_type": self.condition_type,
            "description": description,
            "position_delta": np.array([0, 0]),
            **({"pil_image": [image, condition_img]} if self.return_pil_image else {}),
        }
