import os
import random
import h5py
import numpy as np
import torch
from PIL import Image
from matplotlib.pyplot import imshow, subplot, show
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset, DataLoader
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torchvision import utils

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label



class Randomprocess(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # if random.random() > 0.5:
        #     image, label = random_rot_flip(image, label)
        # elif random.random() > 0.5:
        #     image, label = random_rotate(image, label)
        image = ndimage.rotate(image, -90, order=0, reshape=False)
        label = ndimage.rotate(label, -90, order=0, reshape=False)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))

        # print(image.size(), label.shape)

        sample = {'image': image, 'label': label.long()}
        return sample




class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))

        # print(image.size(), label.shape)

        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train" or self.split == "val":
            slice_name = self.sample_list[idx].strip('\n') # case0031_slice003
            data_path = os.path.join(self.data_dir, slice_name + '.npz') # Synapse/train_npz/case0031_slice003.npz
            data = np.load(data_path)
            image, label = data['image'], data['label']
            # print(type(image), type(label))
            # subplot(121)
            # imshow(image)
            # subplot(122)
            # imshow(label)

        else:
            vol_name = self.sample_list[idx].strip('\n')  # case0008
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name) # Synapse/test_vol_h5/case0008.npy.h5
            data = h5py.File(filepath)  # 加载图片
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample


if __name__ == '__main__':

    db_train = Synapse_dataset(base_dir="D:\\Program Files\\github_code\\project_TransUNet/data/Synapse/train_npz/", list_dir="D:\\Program Files\\github_code\\project_TransUNet\\TransUNet/lists/lists_Synapse/", split="train",
                               transform=None)
    trainloader = DataLoader(db_train, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    for i_batch, sampled_batch in enumerate(trainloader):
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        print(image_batch.shape, label_batch.shape)

        for i in range(8):
            # imshow(utils.make_grid(image_batch[i]))
            # show()
            subplot(4, 4, 2*i+1)
            imshow(image_batch[i])
            subplot(4, 4, 2*i+2)
            imshow(label_batch[i])
        show()
        break

    print("finish ----")
    # data = np.load("D:\\Program Files\\github_code\\project_TransUNet/data/Synapse/train_npz/case0005_slice000.npz")
    # img = data["image"]
    # label = data["label"]
    # print(img.shape, label.shape)
    # subplot(121)
    # # im = Image.fromarray(img)
    # imshow(img)
    # # im.show()
    # # imshow(Image.fromarray(img))
    # subplot(122)
    # imshow(label)
    # show()
