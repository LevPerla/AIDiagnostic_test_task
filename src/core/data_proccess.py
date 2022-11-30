import os

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import scipy
from torch.utils.data import DataLoader

def proccess_data(batch_size=10, train_rete=0.8, val_rate=0.1):
    images = get_images()
    masks = get_masks()

    X = np.array(images, np.float32)
    Y = np.array(masks, np.float32)
    print(f'Loaded {len(X)} images')

    # Split data to train val test
    ix = np.random.choice(len(X), len(X), False)
    tr_ind, val_ind, ts_ind = np.split(ix, [int(len(X) * train_rete), int(len(X) * (train_rete + val_rate))])
    print(len(tr_ind), len(val_ind), len(ts_ind))

    # reshape images
    X = X.reshape(*X.shape, 1)

    # Pack data to batches
    data_tr = DataLoader(list(zip(np.rollaxis(X[tr_ind], 3, 1), Y[tr_ind, np.newaxis])),
                         batch_size=batch_size, shuffle=True)
    data_val = DataLoader(list(zip(np.rollaxis(X[val_ind], 3, 1), Y[val_ind, np.newaxis])),
                          batch_size=batch_size, shuffle=True)
    data_ts = DataLoader(list(zip(np.rollaxis(X[ts_ind], 3, 1), Y[ts_ind, np.newaxis])),
                         batch_size=batch_size, shuffle=True)
    return data_tr, data_val, data_ts


# Read images
def load_dicom(directory):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    image_itk = reader.Execute()

    image_zyx = sitk.GetArrayFromImage(image_itk).astype(np.int16)
    return image_zyx


def get_images():
    print('start image loading')
    # find image dirs
    proccessed_dirs = []
    images = []
    for address, dirs, files in os.walk(os.path.join('..', 'data', 'subset')):
        for name in files:
            dir_path = os.path.join(address, *dirs)
            if dir_path not in proccessed_dirs:
                proccessed_dirs.append(dir_path)
                print(dir_path)
                images_fr_folder = load_dicom(os.path.abspath(dir_path))
                images.append(images_fr_folder)

    images = np.vstack(images)
    print('images uploaded\n')
    return images


def get_masks():
    print('start mask loading')
    # Read masks
    mask = []

    for address, dirs, files in os.walk(os.path.join('..', 'data', 'subset_masks')):
        for name in files:
            print(os.path.join(address, name))
            mask_i = nib.load(os.path.join(address, name))
            mask_i = mask_i.get_fdata().transpose(2, 0, 1)
            mask_i = scipy.ndimage.rotate(mask_i, 90, (1, 2))
            mask.append(mask_i)
    mask = np.vstack(mask)
    print('masks uploaded\n')
    return mask