import glob
import os
import random

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

class ImageDataset(Dataset):
    def __init__(self, root, transform=None, unaligned=False, mode="train", mask=False):
        
        self.transform = transform
        self.unaligned = unaligned
        self.mask = mask

        self.files_A = sorted(glob.glob(os.path.join(root, f"{mode}/A") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, f"{mode}/B") + "/*.*"))
        if mask:
            # Load mask folder
            files_mask = sorted(glob.glob(os.path.join(root, f"{mode}/mask") + "/*.*"))
            # Pair B images with masks by filename stem (mask stem may have a "_mask" suffix).
            def _b_stem(p):
                return os.path.splitext(os.path.basename(p))[0]

            def _mask_stem(p):
                s = os.path.splitext(os.path.basename(p))[0]
                return s[:-5] if s.endswith("_mask") else s

            mask_map = {_mask_stem(p): p for p in files_mask}
            paired_B, paired_mask = [], []
            for fb in self.files_B:
                m = mask_map.get(_b_stem(fb))
                if m is not None:
                    paired_B.append(fb)
                    paired_mask.append(m)
            if not paired_B:
                raise RuntimeError(
                    f"No B/mask filename matches under {root}/{mode}. "
                    "Mask files must share the same stem as the B image "
                    "(optionally with a '_mask' suffix)."
                )
            self.files_B = paired_B
            self.files_B_mask = paired_mask

        # Count number of files for domain A and B
        self.n_files_A = len(self.files_A)
        self.n_files_B = len(self.files_B)

    def __getitem__(self, index):
        if self.unaligned:
            b_index = random.randint(0, len(self.files_B) - 1)
        else:
            b_index = index % len(self.files_B)

        if self.mask:
            # Image and mask
        # Apply transformation
            items = self.transform(
                        image=np.asarray(Image.open(self.files_A[index % len(self.files_A)])),
                        imageB=np.asarray(Image.open(self.files_B[b_index])),
                        maskB=np.asarray(Image.open(self.files_B_mask[b_index]))
            )
            out = {"A": items["image"], "B": items["imageB"], "B_mask": items["maskB"]}
        else:
            # Image only
                    # Apply transformation
            items = self.transform(
                        image=np.asarray(Image.open(self.files_A[index % len(self.files_A)])),
                        imageB=np.asarray(Image.open(self.files_B[b_index]))
                        )
            out = {"A": items["image"], "B": items["imageB"]}
        
        return out                           


    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def num_files_AB(self):
        # Return number of files per domain
        return (self.n_files_A, self.n_files_B)

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert (max_size > 0), "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)
