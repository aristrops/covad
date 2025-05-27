import numpy as np
from PIL import Image
import torch

from torchvision.transforms import transforms
from torch.utils.data.dataset import Dataset


class MvTecConceptDataset(Dataset):
    def __init__(
        self,
        dataframe,
        split: str,
        load_image: bool = True,
        apply_transformation: bool = True,
        img_size=(224, 224),
        use_attr: bool = True,
        n_class_attr: int = 2
    ) -> None:
        super(MvTecConceptDataset)

        self.df = dataframe[dataframe["split"] == split].reset_index(drop=True)

        exclude_cols = ["image_path", "label_index", "mask_path", "split", "path"]
        self.attr_cols = [col for col in self.df.columns if col not in exclude_cols]

        self.split = split
        self.load_image = load_image
        self.apply_transformation = apply_transformation
        self.use_attr = use_attr
        self.n_class_attr = n_class_attr

        transform = transforms.Compose([transforms.Resize(img_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])]) if apply_transformation else None
        
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        label = row["label_index"]

        if self.use_attr:
            attr_label = torch.Tensor(row[self.attr_cols].values.astype(np.float32))
            if self.load_image:
                return image, label, attr_label
            else:
                return attr_label, label
        else:
            return image, label