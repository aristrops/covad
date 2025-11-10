import math
import numpy as np
import torch
from typing import List, Optional
import os
import pandas as pd
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from pathlib import Path
from PIL import Image

from datasets.iad_dataset import IadDataset
from utils.configurations import LabelName

"""Create RealIAD AD samples by parsing the RealIAD AD data file structure.

    The files are expected to follow the structure:
        path/to/dataset/category/OK/Sxxx/image_path.jpg
        path/to/dataset/category/NG/defect_type/Sxxx/image_path.jpg
        path/to/dataset/category/NG/defect_type/Sxxx/mask_path.jpg

    This function creates a dataframe to store the parsed information based on the following format:
"""

DEFECT_TYPES = {
    "AK": "pit",
    "BX": "deformation",
    "CH": "abrasion",
    "HS": "scratch",
    "PS": "damage",
    "QS": "missing parts",
    "YW": "foreign objects",
    "ZW": "contamination",
}

class RealIadDataset(IadDataset):
    def __init__(
        self,
        root: str,
        category: str,
        norm: bool = True,
        img_size=(224, 224),
        gt_mask_size: Optional[tuple] = None,
        preload_imgs: bool = True,
    ) -> None:
        super(RealIadDataset)

        gt_mask_size = img_size if gt_mask_size is None else gt_mask_size

        self.img_size = img_size
        self.gt_mask_size = gt_mask_size

        self.root_category = Path(root) / Path(category)
        self.category = category
        self.samples: pd.DataFrame = None
        self.preload_imgs = preload_imgs

        if norm:
            t_list = [transforms.ToTensor(),
                transforms.Resize(img_size, antialias=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        else:
            t_list = [transforms.ToTensor(),
                transforms.Resize(img_size, antialias=True),
            ]

        self.transform_image = transforms.Compose(t_list)

        self.transform_mask = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    gt_mask_size,
                    antialias=True,
                    interpolation=InterpolationMode.NEAREST,
                ),
            ]
        )

    def contains(self, item) -> bool:
        return self.samples['image_path'].eq(item['image_path']).any()
    
    def load_dataset(self):

        root = Path(self.root_category)

        image_entries = []

        for image_path in root.rglob("*.jpg"):
            parts = image_path.relative_to(root).parts 

            if parts[0] == "NG":
                code = parts[1]
            
            mask_name = image_path.stem + ".png"
            possible_mask_path = image_path.with_name(mask_name)
            if not possible_mask_path.exists():
                label = "good"
                label_index = LabelName.NORMAL
                mask_path = ""

            else:
                label = DEFECT_TYPES.get(code, "unknown")
                label_index = LabelName.ABNORMAL
                mask_path = str(possible_mask_path)


            image_entries.append({
                "image_path": str(image_path),
                "label": label,
                "label_index": label_index,
                "mask_path": mask_path,
                })
            
        if not image_entries:
            raise RuntimeError(f"No valid images found under {root}")
    
        self.samples = pd.DataFrame(image_entries)

        if self.preload_imgs:
            self.data = [self.transform_image(Image.open(row["image_path"]).convert("RGB")
                                              )
                                              for _, row in self.samples.iterrows()
                                              ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):

        row = self.samples.iloc[index]

        # open the image and get the tensor
        if self.preload_imgs:
            image = self.data[index]
        else:
            image = self.transform_image(
                Image.open(row.image_path).convert("RGB")
            )

        label = row.label_index
        path = row.image_path

        if label == LabelName.ABNORMAL and row.mask_path and Path(row.mask_path).exists():
            mask = Image.open(row.mask_path).convert("L")
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros(1, *self.gt_mask_size)

        return image, label, mask.int(), path
    

        