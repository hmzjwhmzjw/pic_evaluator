import torch.utils.data as data

from PIL import Image
import os
import os.path
import random

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MultiLabel(data.Dataset):
    """A generic data loader using a label text like this ::

        dog/xxx.png 0 0 0 0 0
        dog/xxy.png 1 0 0 0 0
        dog/xxz.png 1 1 1 1 1

        cat/123.png 0 0 0 0 1
        cat/nsdf3.png 0 1 0 1 0
        cat/asd932_.png 1 1 0 1 1

    Args:
        root (string): Root directory path of images.
        labelfile(string): Label file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, labelfile, transform=None, target_transform=None,
                 loader=pil_loader):
        lables = open(labelfile,'r')
        imgs = lables.readlines()
        lables.close()
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images!"))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets are multi labels.
        """
        line = self.imgs[index]
        line = line.strip()
        line = line.split()
        path = os.path.join(self.root, line[0])
        targets = []   # here we use the array
        for i in range(1, len(line)):
            targets.append(int(line[i]))

        try:
            img = self.loader(path)
        except:

            random_r = random.randint(0, 255)
            random_g = random.randint(0, 255)
            random_b = random.randint(0, 255)
            if random_r+random_g+random_b>254*3:
                targets = [0, 0, 0, 0, 1]
            else:
                targets = [0, 1, 0, 0, 1]
            img = Image.new('RGB', (400, 400), (random_r, random_g, random_b))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.imgs)
