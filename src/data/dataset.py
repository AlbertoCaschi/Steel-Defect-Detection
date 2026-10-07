import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def rle_decode(rle_str, shape=(256, 1600)):
    """
    Decodifica una stringa RLE in una maschera binaria 2D.
    
    Args:
        rle_str (str o float): Stringa Run-Length Encoding (NaN se assente).
        shape (tuple): Dimensioni dell'immagine (height, width).
        
    Returns:
        np.ndarray: Maschera binaria di tipo uint8 (0 oppure 1).
    """
    if pd.isna(rle_str) or not rle_str or rle_str == "":
        return np.zeros(shape, dtype=np.uint8)
    # no defect --> Mask is full 0s

    height, width = shape
    s = rle_str.split()
    starts = np.asarray(s[0::2], dtype=int) - 1
    lengths = np.asarray(s[1::2], dtype=int)
    ends = starts + lengths
    # save starts, lengths and ends of mask pixels

    mask = np.zeros(height * width, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    # set 1 where there is a mask

    return mask.reshape((height, width), order="F")
    # reshape the mask from a single array to a matrix (W, H) column after column
    # (Fortran order)


class SeverstalDataset(Dataset):
    """
    PyTorch Dataset per la segmentazione semantica dei difetti dell'acciaio Severstal.
    Supporta maschere multi-canale (4 canali per le 4 classi di difetto).
    """
    def __init__(self, df, img_dir, shape=(256, 1600), transform=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame contenente le colonne ['ImageId', 'ClassId', 'EncodedPixels'].
            img_dir (str): Percorso della cartella contenente i file immagine (.jpg).
            shape (tuple): Risoluzione nativa delle immagini (H, W).
            transform (A.Compose, optional): Pipeline di trasformazioni Albumentations.
            is_test (bool): Se True, non carica né restituisce le maschere di ground truth.
        """
        self.img_dir = img_dir
        self.shape = shape
        self.transform = transform
        self.is_test = is_test

        if "ClassId" in df.columns:
            # grouping by images
            self.image_groups = df.groupby("ImageId")
            self.image_ids = list(self.image_groups.groups.keys())
        else:
            # images list
            self.image_ids = df["ImageId"].unique().tolist()
            self.image_groups = None

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.img_dir, image_id)

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Immagine non trovata: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        # TEST -> No assigned defects, no transforms
        if self.is_test:
            if self.transform is not None:
                augmented = self.transform(image=image)
                image = augmented["image"]
            else:
                image = ToTensorV2()(image=image)["image"]
            return {"image": image, "image_id": image_id}

        # TRAIN -> 4 channels (1 binary mask per defect), transforms
        mask = np.zeros((self.shape[0], self.shape[1], 4), dtype=np.float32)

        if self.image_groups is not None and image_id in self.image_groups.groups:
            group = self.image_groups.get_group(image_id)
            for _, row in group.iterrows():
                class_id = int(row["ClassId"])
                rle_pixels = row["EncodedPixels"]
                if pd.notna(rle_pixels):
                    # Assegna la maschera binaria al canale corrispondente (0-indicizzato)
                    mask[:, :, class_id - 1] = rle_decode(rle_pixels, self.shape)

        # transforms (both for image and masks)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.float32).permute(2, 0, 1)
            else:
                mask = mask.permute(2, 0, 1)
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            mask = torch.tensor(mask, dtype=torch.float32).permute(2, 0, 1)

        return {
            "image": image,          # Tensor: (3, H, W)
            "mask": mask,            # Tensor: (4, H, W)
            "image_id": image_id
        }


def get_transforms(phase="train"):
    """
    Transforms pipeline for both train and val sets
    """
    if phase == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            A.Normalize(
                # ImageNet statistics (for the unet model used)
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(
                # ImageNet statistics
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])