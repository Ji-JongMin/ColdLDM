import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import os
import numpy as np
import random
import pydicom
from numpy.ma import masked_array
import cv2
import random as random2

def cycle(dl):
    """Utility function to create an infinite data loader."""
    while True:
        for data1, data2 in dl:
            yield data1, data2


class Dataset(Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.folder = data_root
        self.full = []
        self.qut = []
        self.file = sorted(os.listdir(self.folder))[:7]
        for idx in np.arange(len(self.file)):
            full_name = sorted(os.listdir(os.path.join(self.folder, self.file[idx], "full_1mm")))
            qut_name = sorted(os.listdir(os.path.join(self.folder, self.file[idx], "quarter_1mm")))
            self.full.extend([os.path.join(self.folder, self.file[idx], "full_1mm", fn) for fn in full_name])
            self.qut.extend([os.path.join(self.folder, self.file[idx], "quarter_1mm", qn) for qn in qut_name])

        self.transform = T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.full)

    def window_image(self, img, rescale=True):
        img_mean, img_std = np.mean(img), np.std(img)
        return (img - img_mean) / img_std if rescale else img
        
    def adjust_contrast(self,x):
        gamma = (round(random2.uniform(0.8,1.2),1))
        min = np.min(x)
        max = np.max(x)
        intensity_range = max - min
        x = ((x - min) / intensity_range) ** gamma * intensity_range + min
        return x 
        
    def make_mask(self, img):
        mask = masked_array(data=img, mask=img < np.max(img) * 0.22)
        binary_mask = (~mask.mask).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(binary_mask)
        cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
        mask //= 255
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        output_img = np.zeros_like(img)
        output_img[labels == largest_component] = 1
        return output_img
    
    def random_crop_brain_area(self, ct_image1, ct_image2, mask, crop_size=(64, 64)):
        indices = np.where(mask != 0)
        if not len(indices[0]):
            raise ValueError("No brain area found in the mask.")
        rows, cols = indices
        idx = random.choice(range(len(rows)))
        start_row, start_col = rows[idx], cols[idx]
        return (ct_image1[start_row:start_row+crop_size[0], start_col:start_col+crop_size[1]],
                ct_image2[start_row:start_row+crop_size[0], start_col:start_col+crop_size[1]])

    def __getitem__(self, index):
        data1 = pydicom.read_file(self.full[index]).pixel_array
        data2 = pydicom.read_file(self.qut[index]).pixel_array
        mask = self.make_mask(data1)
        data1, data2 = data1 * mask, data2 * mask
        img1, img2 = self.window_image(data1), self.window_image(data2)
        return self.transform(img1), self.transform(img2)

