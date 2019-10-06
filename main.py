import os
import numpy as np
import matplotlib.pyplot as plt

from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import Input, Conv3D, Conv3DTranspose, MaxPooling3D, concatenate, Dropout
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

import tensorflow as tf
# import tensorflow_datasets

import pydicom as dicom
import dicom_numpy as dn
import SimpleITK as sitk
from mayavi import mlab

start_neurons = 16


# dice = 2*intersection/union
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


# Index de Jaccard = intersection/union; distance de Jaccard = 1-index
def jaccard_distance_loss(y_true, y_pred):
    smooth = 100.
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def bce_jaccard_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)+jaccard_distance_loss(y_true, y_pred)


def dice_metric(label, pred):
    return tf.py_func(dice_loss, [label, pred], tf.float64)


def build_model(input_layer, start_neurons):
    conv1 = Conv3D(start_neurons * 1, (3, 3, 3), activation="relu", padding="same")(input_layer)
    conv1 = Conv3D(start_neurons * 1, (3, 3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling3D((2, 2, 2), padding="same")(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv3D(start_neurons * 2, (3, 3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv3D(start_neurons * 2, (3, 3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling3D((2, 2, 2), padding="same")(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv3D(start_neurons * 4, (3, 3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv3D(start_neurons * 4, (3, 3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling3D((2, 2, 2), padding="same")(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv3D(start_neurons * 8, (3, 3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv3D(start_neurons * 8, (3, 3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling3D((2, 2, 2), padding="same")(conv4)
    pool4 = Dropout(0.5)(pool4)

    # Middle
    convm = Conv3D(start_neurons * 16, (3, 3, 3), activation="relu", padding="same")(pool4)
    convm = Conv3D(start_neurons * 16, (3, 3, 3), activation="relu", padding="same")(convm)

    deconv4 = Conv3DTranspose(start_neurons * 8, (3, 3, 3), strides=(2, 2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv3D(start_neurons * 8, (3, 3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv3D(start_neurons * 8, (3, 3, 3), activation="relu", padding="same")(uconv4)

    deconv3 = Conv3DTranspose(start_neurons * 4, (3, 3, 3), strides=(2, 2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv3D(start_neurons * 4, (3, 3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv3D(start_neurons * 4, (3, 3, 3), activation="relu", padding="same")(uconv3)

    deconv2 = Conv3DTranspose(start_neurons * 2, (3, 3, 3), strides=(2, 2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv3D(start_neurons * 2, (3, 3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv3D(start_neurons * 2, (3, 3, 3), activation="relu", padding="same")(uconv2)

    deconv1 = Conv3DTranspose(start_neurons * 1, (3, 3, 3), strides=(2, 2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv3D(start_neurons * 1, (3, 3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv3D(start_neurons * 1, (3, 3, 3), activation="relu", padding="same")(uconv1)

    uconv1 = Dropout(0.5)(uconv1)
    output_layer = Conv3D(1, (1, 1, 1), padding="same", activation="sigmoid")(uconv1)

    return output_layer


def extract_voxel_data(DCM_files):
    datasets = [dicom.read_file(f) for f in DCM_files]
    try:
        voxel_ndarray, ijk_to_xyz = dn.combine_slices(datasets)
    except dn.DicomImportException as e:
        raise e
    return voxel_ndarray


def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing


# telechargement de l'image (DICOM -> numpy)
PathDicom = "./DICOM/1/"
DCM_files = []
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():
            DCM_files.append(os.path.join(dirName, filename))

PathLabels = "./Label/"
label_files = []
for dirName, subdirList, fileList in os.walk(PathLabels):
    for filename in fileList:
        if ".mhd" in filename.lower():
            label_files.append(os.path.join(dirName, filename))

train_x = extract_voxel_data(DCM_files)

# chaque element de train_y represente 1 cube de 256*256*136 + metadonnees
train_y = [load_itk(label)[0] for label in label_files]

# Complementer les y par les zeros
# train_y = np.array(
#     [np.concatenate((i, np.zeros((i.shape[0], i.shape[1], train_x.shape[2] - i.shape[2]))), axis=2) for i in train_y])
train_y = np.array(train_y)
train_x = np.array([train_x])

train_x = train_x[:, :, :, 58:186]
train_y = train_y[:, :, :, 4:132]

enter_shape = train_x.shape + (1,)
train_x = train_x.reshape(*enter_shape)
train_y = train_y.reshape(*enter_shape)

# Visualisation
# train_y = np.array(train_y[0])
# print(train_x.shape)
# mlab.contour3d(train_y)
# mlab.savefig('surface.obj')
# input()
#

input_layer = Input(enter_shape[1:])
output_layer = build_model(input_layer, start_neurons)
model = Model(input_layer, output_layer)
model.compile(loss=bce_jaccard_loss, optimizer=Adam(lr=1e-3), metrics=[dice_coef])
model.save_weights('./keras.weights')

history = model.fit(train_x, train_y,
                    batch_size=1,
                    epochs=1,
                    verbose=1,
                    validation_split=1)
