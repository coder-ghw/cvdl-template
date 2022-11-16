from torchvision import datasets, transforms
import os
import sys
from skimage import io, transform
import numpy as np
import torch
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from base import BaseDataLoader


# 1.####################################MnistDataLoader########################################

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self,
                 data_dir,
                 batch_size,
                 shuffle=True,
                 validation_split=0.0,
                 num_workers=1,
                 training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir,
                                      train=training,
                                      download=True,
                                      transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split,
                         num_workers)


# 2.####################################FaceLandmarksDataset########################################

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]
        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top:top + new_h, left:left + new_w]
        landmarks = landmarks - [left, top]
        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {
            'image': torch.from_numpy(image),
            'landmarks': torch.from_numpy(landmarks)
        }


class FaceLandmarksDataset(BaseDataLoader):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transfm=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transfm

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx,
                                                                         0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    composed = transforms.Compose([Rescale(256),
                                   RandomCrop(224)])
    face_data = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                     root_dir='data/faces/',
                                     transfm=composed)
    for i in range(len(face_data)):
        sample = face_data[i]
        print("sample['image'].shape: {}".format(sample['image'].shape))
        print("sample['landmarks'].shape: {}".format(sample['landmarks'].shape))
