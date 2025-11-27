import os
import glob
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from app.data_generator import LandslideDataGenerator
from app.model_utils import dice_loss, dice_coefficient, f1_m, precision_m, recall_m
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, UpSampling2D, AveragePooling2D, Conv2DTranspose, Concatenate, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

# U-Net Model
def unet_model(input_size=(128, 128, 6)):
    inputs = Input(input_size)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = Concatenate()([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = Concatenate()([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = Concatenate()([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = Concatenate()([u9, c1])
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def ASPP(inputs):
    shape = inputs.shape
    y_pool = AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)
    y_pool = Conv2D(256, 1, padding='same', use_bias=False)(y_pool)
    y_pool = BatchNormalization()(y_pool)
    y_pool = Activation('relu')(y_pool)
    y_pool = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y_pool)
    y_1 = Conv2D(256, 1, dilation_rate=1, padding='same', use_bias=False)(inputs)
    y_1 = BatchNormalization()(y_1)
    y_1 = Activation('relu')(y_1)
    y_6 = Conv2D(256, 3, dilation_rate=6, padding='same', use_bias=False)(inputs)
    y_6 = BatchNormalization()(y_6)
    y_6 = Activation('relu')(y_6)
    y_12 = Conv2D(256, 3, dilation_rate=12, padding='same', use_bias=False)(inputs)
    y_12 = BatchNormalization()(y_12)
    y_12 = Activation('relu')(y_12)
    y_18 = Conv2D(256, 3, dilation_rate=18, padding='same', use_bias=False)(inputs)
    y_18 = BatchNormalization()(y_18)
    y_18 = Activation('relu')(y_18)
    y = Concatenate()([y_pool, y_1, y_6, y_12, y_18])
    y = Conv2D(256, 1, dilation_rate=1, padding='same', use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    return y

def DeepLabV3Plus(input_shape=(128, 128, 6)):
    inputs = Input(input_shape)
    # Use only first 3 channels for ResNet50 (RGB)
    rgb_inputs = tf.keras.layers.Lambda(lambda x: x[..., :3])(inputs)
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=rgb_inputs)
    image_features = base_model.get_layer('conv4_block6_out').output
    x_a = ASPP(image_features)
    x_a = UpSampling2D((4, 4), interpolation="bilinear")(x_a)
    x_b = base_model.get_layer('conv2_block2_out').output
    x_b = Conv2D(48, 1, padding='same', use_bias=False)(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Activation('relu')(x_b)
    x = Concatenate()([x_a, x_b])
    x = Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((4, 4), interpolation="bilinear")(x)
    x = Conv2D(1, (1, 1))(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=inputs, outputs=x)
    return model

def prepare_dataset():
    data_dir = './your_data_root_dir'  # Change this to your dataset root
    if not os.path.exists(os.path.join(data_dir, 'TrainData')):
        if os.path.exists(os.path.join(data_dir, 'Landslide4Sense')):
            data_dir = os.path.join(data_dir, 'Landslide4Sense')
        elif os.path.exists(os.path.join(data_dir, 'landslide4sense')):
            data_dir = os.path.join(data_dir, 'landslide4sense')
        else:
            raise FileNotFoundError("Could not find TrainData directory in dataset")
    return data_dir

def train_unet(train_img_paths, train_mask_paths):
    train_img, val_img, train_mask, val_mask = train_test_split(
        train_img_paths, train_mask_paths, test_size=0.2, random_state=42)
    train_gen = LandslideDataGenerator(train_img, train_mask, batch_size=8, model_type='unet')
    val_gen = LandslideDataGenerator(val_img, val_mask, batch_size=8, shuffle=False, model_type='unet')
    callbacks = [
        EarlyStopping(patience=10, monitor='val_dice_coefficient', mode='max', restore_best_weights=True),
        ModelCheckpoint('models/landslide_unet_model.h5', monitor='val_dice_coefficient', mode='max', save_best_only=True)
    ]
    model = unet_model()
    model.compile(optimizer=Adam(1e-4), loss=dice_loss, metrics=['accuracy', dice_coefficient, f1_m, precision_m, recall_m])
    model.fit(train_gen, validation_data=val_gen, epochs=50, callbacks=callbacks)
    return model

def train_deeplab(train_img_paths, train_mask_paths):
    train_img, val_img, train_mask, val_mask = train_test_split(
        train_img_paths, train_mask_paths, test_size=0.2, random_state=42)
    train_gen = LandslideDataGenerator(train_img, train_mask, batch_size=8, model_type='deeplab')
    val_gen = LandslideDataGenerator(val_img, val_mask, batch_size=8, shuffle=False, model_type='deeplab')
    callbacks = [
        EarlyStopping(patience=10, monitor='val_dice_coefficient', mode='max', restore_best_weights=True),
        ModelCheckpoint('models/landslide_deeplab_model.h5', monitor='val_dice_coefficient', mode='max', save_best_only=True)
    ]
    model = DeepLabV3Plus()
    model.compile(optimizer=Adam(1e-4), loss=dice_loss, metrics=['accuracy', dice_coefficient, f1_m, precision_m, recall_m])
    model.fit(train_gen, validation_data=val_gen, epochs=50, callbacks=callbacks)
    return model

def main():
    data_dir = prepare_dataset()
    train_img_dir = os.path.join(data_dir, 'TrainData', 'img')
    train_mask_dir = os.path.join(data_dir, 'TrainData', 'mask')
    train_img_paths = sorted(glob.glob(os.path.join(train_img_dir, '*.h5')))
    train_mask_paths = sorted(glob.glob(os.path.join(train_mask_dir, '*.h5')))
    if len(train_img_paths) == 0:
        train_img_paths = sorted(glob.glob(os.path.join(train_img_dir, '*img.h5')))
        train_mask_paths = sorted(glob.glob(os.path.join(train_mask_dir, '*mask.h5')))
    assert len(train_img_paths) == len(train_mask_paths), "Mismatch between number of images and masks"
    print(f"Found {len(train_img_paths)} images and {len(train_mask_paths)} masks")
    print("Training U-Net...")
    unet_model_obj = train_unet(train_img_paths, train_mask_paths)
    print("Training DeepLabV3+...")
    deeplab_model_obj = train_deeplab(train_img_paths, train_mask_paths)
    print("Training complete. Models are saved in the 'models/' directory.")

if __name__ == "__main__":
    main()
