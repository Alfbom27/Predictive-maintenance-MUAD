from torch.utils.data import Dataset
import torchvision.transforms as T
import os
from PIL import Image
import torch


class MIADDataset(Dataset):
    def __init__(self, dataset_path=None, mode="train", class_list=None, transform=None, gt_transform=None, img_size=448, crop_size=392, kd_training=False):
        self.dataset_path = dataset_path
        self.mode = mode
        self.transform = transform
        self.gt_transform = gt_transform
        self.img_size = img_size
        self.crop_size = crop_size
        self.class_list = class_list
        self.kd_training = kd_training

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
        if self.kd_training:
            return img
        img = self.transform(img)

        if self.labels[idx] == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(self.gt_paths[idx])
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], f"Shape mismatch: Img: {img.size()}, gt: {gt.size()}"

        return img, gt, self.labels[idx]


class BSDataDataset(Dataset):
    def __init__(self, dataset_path=None, mode="train", class_list=None, transform=None, gt_transform=None, augmentations=False):
        self.dataset_path = dataset_path
        self.mode = mode
        self.transform = transform
        self.gt_transform = gt_transform
        self.class_list = class_list
        self.augmentations = augmentations

        if self.transform is None:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Resize((392, 965)),
                T.CenterCrop((392, 784)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        if self.gt_transform is None:
            self.gt_transform = T.Compose([
                T.ToTensor(),
                T.Resize((392, 965)),
                T.CenterCrop((392, 784)),
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
            gt = torch.zeros([1, img.size()[-2], img.size()[-1]])
        else:
            gt = Image.open(self.gt_paths[idx])
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], f"Shape mismatch: Img: {img.size()}, gt: {gt.size()}"

        return img, gt, self.labels[idx]


class KDTransforms:
    def __init__(self):
        self.global_transform = T.Compose([
            T.RandomResizedCrop(224, scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.2, 0.1),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(23, sigma=(0.1, 2.0)),
            T.ToTensor(),
        ])

        self.local_transform = T.Compose([
            T.RandomResizedCrop(98, scale=(0.05, 0.4)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.2, 0.1),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(7, sigma=(0.1, 2.0)),
            T.ToTensor(),
        ])

    def __call__(self, img):
        global_crops = [self.global_transform(img) for _ in range(2)]
        local_crops = [self.local_transform(img) for _ in range(8)]
        return global_crops, local_crops


class KDdataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transforms = KDTransforms()

    def __getitem__(self, idx):
        img = self.dataset[idx]
        g, l = self.transforms(img)

        views = g + l
        return views

    def __len__(self):
        return len(self.dataset)


