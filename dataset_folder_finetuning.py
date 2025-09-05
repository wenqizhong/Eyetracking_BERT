# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Modified on torchvision code bases
# https://github.com/pytorch/vision
# --------------------------------------------------------'
from torchvision.datasets.vision import VisionDataset

from PIL import Image

import os
import os.path
import re
import random
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions) 


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def check_column(file_path, target_string, img_size):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            x_column_values = []
            y_column_values = []
            img_column_values = []
            position_pairs = []
            x_limited = 240/(256/img_size[0])
            y_limited = 240/(256/img_size[1])

            for line in lines:
                columns = line.strip().split()
                x_column_values.append(float(columns[1]))  # Convert x to float
                y_column_values.append(float(columns[2]))  # Convert y to float
                img_column_values.append(columns[-1])

            img = ' '.join(img_column_values)
            
            pattern = re.compile(r'\b{}\b'.format(re.escape(target_string)))
            occurrences = len(re.findall(pattern, img))

            if occurrences > 0:
                # Find indices of occurrences
                indices = [i for i, value in enumerate(img_column_values) if re.search(pattern, value)]
                
                # Output corresponding x and y values for each occurrence
                for index in indices:
                    x_value = x_column_values[index]
                    y_value = y_column_values[index]
                    if 0 < x_value <= x_limited and 0 < y_value <= y_limited:
                        position_pairs.append((x_value, y_value))
            else:
                position_pairs = None
                
            return occurrences, position_pairs

    except FileNotFoundError:
        return False

   

def  last_folder(path):
    path = path.split('/')
    return path[-1]

def scale_position_fixations(position_fixations, orig_size, process_size, final_size):
    scaled_positions = []
    cropped_positions = []
    # scaling
    if position_fixations is None or position_fixations is []:
        return None
    for (x, y) in position_fixations:
        x_scale = process_size[0] / orig_size[0]
        y_scale = process_size[1] / orig_size[1]
        
        scaled_positions = [(x * x_scale, y * y_scale) for (x, y) in position_fixations]

        offset_x = 16
        offset_y = 16
        # offset_x = (process_size[0] - final_size[0]) / 2 if process_size[0] > final_size[0] else 0
        # offset_y = (process_size[1] - final_size[1]) / 2 if process_size[1] > final_size[1] else 0

        cropped_positions = [
            (x - offset_x if x >= 16 else x, y - offset_y if y >= 16 else y)
            for (x, y) in scaled_positions
        ] 

    return cropped_positions

def get_patch_indices(position_fixations, img_size, patch_size):
    if position_fixations is None or position_fixations is []:
        return None
    patch_indices = []
    for (x, y) in position_fixations:
        i = y // patch_size
        # j = x // patch_size + 1
        if x != 224:
            j = x // patch_size + 1
        else:
            j = x // patch_size
        if i != 14:
            patch_idx = i * (img_size // patch_size) + j
        else:
            patch_idx = (i - 1) * (img_size // patch_size) + j
        patch_indices.append(patch_idx)
    return patch_indices

def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file) 
    for target_class in sorted(class_to_idx.keys()): #target_class: 'val', 'train'
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)): 
            if last_folder(root) == 'image':
                print(root) 
                txt_dir_path = os.path.join(os.path.dirname(root), 'data')
                for fname in sorted(fnames):  #fname: 'ILSVRC2012_train_xxxx.JPEG', ...
                    for txt in sorted(os.listdir(txt_dir_path)):
                        txt_idx, _ = os.path.splitext(txt)
                        sub_idx = int(txt_idx)
                        txt_path = os.path.join(txt_dir_path, txt)
                        path = os.path.join(root, fname)
                        img = Image.open(path)  
                        fixations = check_column(txt_path, fname.split('.')[0], img.size)
                        position_fixations = fixations[1]
                        if position_fixations is not None:
                            if  len(position_fixations) >= 15:
                                position_fixations = position_fixations[:15]
                            elif  0 < len(position_fixations) < 15:
                                padding_length = 15 - len(position_fixations)
                                position_fixations.extend([(0, 0)] * padding_length)
                        position_fixations = scale_position_fixations(position_fixations, img.size, (256, 256), (224, 224))
                        patch_indices = get_patch_indices(position_fixations, 224, 16)                      
                        if is_valid_file(path):
                            item = path, patch_indices, class_index, sub_idx
                            if item[1] is not None and item[1] != []:
                                instances.append(item)
                            # else:
                            #     print(1)
                                
                
    return instances



class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of ((sample path, class_index), num_fixations, position_fixations) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(    
            self,
            root: str,
            data_type: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,  
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)  
        # self.data_type = data_type
        classes, class_to_idx = self._find_classes(data_type)  
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file) 
        # print(samples)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.patch_indices = [s[1] for s in samples]
        self.targets = [s[2] for s in samples]
        self.sub_idx = [s[3] for s in samples]

     
 
    def _find_classes(self, data_type):
        datasets_classes = {
            'dataset_ASD': {
                'classes': ['TD', 'ASD'],
                'class_to_idx': {'TD': 0, 'ASD': 1}
            },
            'dataset_age': {
                'classes': ['18mos', '30mos'],
                'class_to_idx': {'18mos': 0, '30mos': 1}
            },
            'dataset_task_1': {
            'classes': ['free', 'object'],
            'class_to_idx': {'free': 0, 'object': 1}
            },
            'dataset_task_2': {
            'classes': ['free', 'saliency'],
            'class_to_idx': {'free': 0, 'saliency': 1}
            },
            'dataset_gender': {
            'classes': ['female', 'male'],
            'class_to_idx': {'female': 0, 'male': 1},
            }
        }
        config = datasets_classes.get(data_type)
        if config is not None:
            return config['classes'], config['class_to_idx']
        else:
            raise ValueError(f"Dataset '{data_type}' not found in configuration.")   

    # def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
    #     """
    #     Finds the class folders in a dataset.

    #     Args:
    #         dir (string): Root directory path.

    #     Returns:
    #         tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

    #     Ensures:
    #         No class is a subdirectory of another.
    #     """
    #     # classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    #     # classes.sort()   #classes:['val', 'train']
    #     # class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}  #class_to_idx:{'val': 0, 'train': 1}
    #     #ASD,TD分类
    #     classes = ['TD', 'ASD']  
    #     class_to_idx = {'TD': 0, 'ASD': 1}

    #     #18,30年龄分类
    #     classes = ['18mos', '30mos']  
    #     class_to_idx = {'18mos': 0, '30mos': 1}
    #     return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, position_fixations, target) where target is class_index of the target class.
        """
        while True:
            try:
                path, patch_indices, target, sub_idx = self.samples[index]
                sample = self.loader(path)
                break
            except Exception as e:
                print(e)
                index = random.randint(0, len(self.samples) - 1)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)


        return sample, patch_indices, target, sub_idx

    def __len__(self) -> int:
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            data_type: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ImageFolder, self).__init__(root, data_type, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples
        self.data_type = data_type


# data_type = 'dataset_ASD',
# dataset = ImageFolder('./dataset_G1+G2_cross/group1/train', data_type = 'dataset_ASD', transform=None)


