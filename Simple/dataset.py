''' Datasets
    This file contains definitions for our CIFAR, ImageFolder, and HDF5 datasets
'''
import os
import sys
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import re

import torchvision.transforms as transforms
import torch.utils.data as data
         

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
  with open(path, 'rb') as f:
    img = Image.open(f)
    return img.convert('RGB')


class MagicTransform(object):

  def __init__(self, custom_transforms):
    self.custom_transforms = custom_transforms

  def __call__(self, img):
    """
    Args:
        img (PIL Image): Image to be Magic transformed.
    Returns:
        PIL Image: Pasted image.
    """
    original_img = img
    img = transforms.functional.crop(img, 70, 30, 310, 421)
    for custom_transform in self.custom_transforms:
      img = custom_transform
    return original_img.paste(img, (30, 70))

  def __repr__(self):
    return self.__class__.__name__


class MagicDataset(data.Dataset):
  """A data loader for the Magic Dataset."""
  def __init__(self, root, transform=None,
               loader=pil_loader, load_in_mem=True, **kwargs):
    METADATA_FILE = "AllCards.json"
    DATASET_FOLDERS_FILE = "dataset_folders"
    metadata = json.load(open(os.path.join(root, METADATA_FILE)))
    dataset_folders = open(os.path.join(root, DATASET_FOLDERS_FILE))
    # data_dict = {}
    # classes_dict = {}
    classes = {}
    card_to_name = {}
    name_to_class = {}

    #iterate for classes
    for folder in dataset_folders:
      folder = os.path.join(root, folder.strip())
      for card_filename in os.listdir(folder):
          card_name = re.split("( \[.*\])?\.", card_filename)[0]
          try:
            if (("Creature" in metadata[card_name]["type"] or 
                "Land" in metadata[card_name]["type"])
                and metadata[card_name]["subtypes"]):
                classes.setdefault(metadata[card_name]["subtypes"][0], 0)
                classes[metadata[card_name]["subtypes"][0]] += 1
          except:
            continue

    classes_with_occurence = sorted([(k, v) for k, v in classes.items()], key= lambda i: i[1], reverse=True)
    classes = [x for x, y in classes_with_occurence]
    #classes = classes[:16]
    # classes = ['Forest', 'Island']
    dataset_folders = open(os.path.join(root, DATASET_FOLDERS_FILE))
    # iterate for images
    for folder in dataset_folders:
      folder = os.path.join(root, folder.strip())
      for card_filename in os.listdir(folder):
          card_name = re.split("( \[.*\])?\.", card_filename)[0]
          try:
            if (("Creature" in metadata[card_name]["type"] or 
                "Land" in metadata[card_name]["type"])
                and metadata[card_name]["subtypes"] and 
                metadata[card_name]["subtypes"][0] in classes):
                
                card_path = os.path.join(folder, card_filename)
                # if card_name in data_dict.keys():
                #   data_dict[card_name].append(card_path)
                # else:
                #   data_dict[card_name] = [card_path]
                card_to_name[card_path] = card_name
                name_to_class[card_name] = metadata[card_name]["subtypes"][0]
                #imgs.append((card_path, ))
          except:
            continue

    # for k, v in data_dict.items():
    #   k2 = metadata[k]["subtypes"][0]
    #   if k2 in classes_dict.keys():
    #       classes_dict[k2].extend(v)
    #   else:
    #       classes_dict[k2] = v.copy()
    class_to_idx = {k: i for i, k in enumerate(classes)}
    #class_to_idx = {k: i for i, k in enumerate(set(name_to_class.values()))}
    imgs = {i: (card_path, class_to_idx[name_to_class[card_to_name[card_path]]]) 
            for i, card_path in enumerate(card_to_name)}
    print(len(imgs))
    self.root = root
    self.class_to_idx = class_to_idx
    self.classes = classes
    self.imgs = imgs
    self.transform = transform
    self.loader = loader
    self.load_in_mem = load_in_mem
    # self.data_dict = data_dict
    # self.classes_dict = classes_dict
    if self.load_in_mem:
      print('Loading all images into memory...')
      self.data, self.labels = [], []
      for index in tqdm(range(len(self.imgs))):
        path, target = imgs[index][0], imgs[index][1]
        img = self.transform(self.loader(path))
        self.data.append(img)
        # plt.imshow(torch.transpose(img, 0, 2))
        # plt.show()
        self.labels.append(target)


  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is class_index of the target class.
    """
    if self.load_in_mem:
        img = self.data[index]
        target = self.labels[index]
    else:
      path, target = self.imgs[index]
      img = self.loader(str(path))
      if self.transform is not None:
        img = self.transform(img)
    return img, int(target)

  def __len__(self):
    return len(self.imgs)

  def __repr__(self):
    fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
    fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
    fmt_str += '    Root Location: {}\n'.format(self.root)
    tmp = '    Transforms (if any): '
    fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    return fmt_str
        

