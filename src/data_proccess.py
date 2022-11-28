import SimpleITK as sitk
import nibabel as nib
import os
import numpy as np
import scipy
from skimage.transform import resize
from torch.utils.data import DataLoader

def proccess_data(batch_size=10):
    images = get_images()
    mask = get_masks()
    size = (512, 512)
    X = [resize(x, size, mode='constant', anti_aliasing=True) for x in images]
    Y = [resize(y, size, mode='constant', anti_aliasing=False) for y in mask]

    X = np.array(X, np.float32)
    Y = np.array(Y, np.float32)
    print(f'Loaded {len(X)} images')

    ix = np.random.choice(len(X), len(X), False)
    tr, val, ts = np.split(ix, [int(len(X) * 0.8), int(len(X) * 0.9)])
    print(len(tr), len(val), len(ts))

    X = X.reshape(-1, *size, 1)

    data_tr = DataLoader(list(zip(np.rollaxis(X[tr], 3, 1), Y[tr, np.newaxis])),
                         batch_size=batch_size, shuffle=True)
    data_val = DataLoader(list(zip(np.rollaxis(X[val], 3, 1), Y[val, np.newaxis])),
                          batch_size=batch_size, shuffle=True)
    data_ts = DataLoader(list(zip(np.rollaxis(X[ts], 3, 1), Y[ts, np.newaxis])),
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
    # find image dirs
    proccessed_dirs = []
    images = []

    for address, dirs, files in os.walk('subset_img/subset'):
        for name in files:
            dir_path = os.path.join(address, *dirs)
            if dir_path not in proccessed_dirs and '.ipynb_checkpoints' not in dir_path:
                proccessed_dirs.append(dir_path)
                print(dir_path)
                images_fr_folder = load_dicom(dir_path)
                images.append(images_fr_folder)

    images = np.vstack(images)
    return images

def get_masks():
    # Read masks
    mask = []

    for address, dirs, files in os.walk('subset_masks'):
        for name in files:
            print(os.path.join(address, name))
            mask_i = nib.load(os.path.join(address, name))
            mask_i = mask_i.get_fdata().transpose(2, 0, 1)
            mask_i = scipy.ndimage.rotate(mask_i, 90, (1, 2))
            mask.append(mask_i)
    mask = np.vstack(mask)
    return mask