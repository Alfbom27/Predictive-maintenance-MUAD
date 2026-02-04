from torch.utils.data import Dataset
import torchvision.transforms as T
import os
from PIL import Image
import torch


class MIADDataset(Dataset):
    def __init__(self, dataset_path=None, mode="train", class_list=None, transform=None, gt_transform=None, img_size=448, crop_size=392):
        self.dataset_path = dataset_path
        self.mode = mode
        self.transform = transform
        self.gt_transform = gt_transform
        self.img_size = img_size
        self.crop_size = crop_size
        self.class_list = class_list

        if self.transform is None:
            self.transform = T.Compose([
                T.Resize(self.img_size),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        if self.gt_transform is None:
            self.gt_transform = T.Compose([
                T.Resize(self.img_size),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
            ])

        self.image_paths = []
        self.labels = []
        self.gt_paths = []


        if mode == "train" and dataset_path is not None:
            for image_class in self.class_list:
                class_dir = os.path.join(self.dataset_path, image_class, "train", "good")
                for filename in os.listdir(class_dir):
                    self.image_paths.append(os.path.join(class_dir, filename))
                    self.gt_paths.append(0)
                    self.labels.append(0)

        elif mode == "test" and dataset_path is not None:
            for image_class in self.class_list:
                class_dir = os.path.join(self.dataset_path, image_class, "test")
                for type_dir in os.listdir(class_dir):
                    if type_dir == "good":
                        for filename in os.listdir(os.path.join(class_dir, type_dir)):
                            self.image_paths.append(os.path.join(class_dir, type_dir, filename))
                            self.gt_paths.append(0)
                            self.labels.append(0)
                    else:
                        for filename in os.listdir(os.path.join(class_dir, type_dir)):
                            self.image_paths.append(os.path.join(class_dir, type_dir, filename))
                            self.labels.append(1)
                            name, ext = os.path.splitext(filename)
                            mask_name = name + "_mask" + ext
                            self.gt_paths.append(os.path.join(self.dataset_path, image_class, "ground_truth", type_dir, mask_name))

        assert len(self.image_paths) == len(self.labels), "Mismatch between image paths and labels"


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)

        if self.labels[idx] == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(self.gt_paths[idx])
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], f"Shape mismatch: Img: {img.size()}, gt: {gt.size()}"

        return img, gt, self.labels[idx]







