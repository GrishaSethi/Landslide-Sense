import numpy as np
import h5py
from tensorflow.keras.utils import Sequence

class LandslideDataGenerator(Sequence):
    def __init__(self, img_paths, mask_paths, batch_size=16, shuffle=True, model_type='unet'):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.model_type = model_type
        self.indexes = np.arange(len(self.img_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.img_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_img_paths = [self.img_paths[i] for i in batch_indexes]
        batch_mask_paths = [self.mask_paths[i] for i in batch_indexes]
        X, y = self.__data_generation(batch_img_paths, batch_mask_paths)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_img_paths, batch_mask_paths):
        X = np.zeros((len(batch_img_paths), 128, 128, 6), dtype=np.float32)
        y = np.zeros((len(batch_mask_paths), 128, 128, 1), dtype=np.float32)
        for i, (img_path, mask_path) in enumerate(zip(batch_img_paths, batch_mask_paths)):
            with h5py.File(img_path, 'r') as hdf:
                data = np.array(hdf.get('img'), dtype=np.float32)
                data[np.isnan(data)] = 0.000001
                data_red = data[:, :, 3] / 255.0
                data_green = data[:, :, 2] / 255.0
                data_blue = data[:, :, 1] / 255.0
                data_nir = data[:, :, 7] / 255.0
                data_slope = data[:, :, 12] / 90.0
                data_elevation = data[:, :, 13] / 5000.0
                data_ndvi = np.divide(data_nir - data_red, data_nir + data_red + 1e-6)
                X[i, :, :, 0] = data_red
                X[i, :, :, 1] = data_green
                X[i, :, :, 2] = data_blue
                X[i, :, :, 3] = data_ndvi
                X[i, :, :, 4] = data_slope
                X[i, :, :, 5] = data_elevation
            with h5py.File(mask_path, 'r') as hdf:
                mask = np.array(hdf.get('mask'), dtype=np.float32)
                mask[np.isnan(mask)] = 0.0
                y[i, :, :, 0] = mask
        return X, y
