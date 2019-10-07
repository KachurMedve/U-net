import os
import numpy as np

from keras import backend as K
from keras.losses import binary_crossentropy

import pydicom as dicom
import dicom_numpy as dn
import SimpleITK as sitk
from mayavi import mlab

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
import torch.utils.data as data

from torchvision import datasets, transforms

start_neurons = 16
num_epochs = 1
batch_size = 1
lr = 0.001


def Down(k1, k2, do=0.5):
    return nn.Sequential(
        nn.Conv3d(k1, k2, 3),
        nn.ReLU(True),
        nn.Conv3d(k2, k2, 3),
        nn.ReLU(True),
        nn.MaxPool3d(2),
        nn.Dropout3d(do),
    )


class Up(nn.Module):
    def __init__(self, k1, k2):
        super(Up, self).__init__()

        self.deconv = nn.ConvTranspose3d(k1, k2, 3, 2)
        self.double_conv = nn.Sequential(
            nn.Dropout3d(0.5),
            nn.Conv3d(k1, k2, 3),
            nn.ReLU(True),
            nn.Conv3d(k2, k2, 3),
            nn.ReLU(True),
        )

    @staticmethod
    def crop_centre(layer, target_size, diff):
        diff //= 2
        return layer[:, :, diff[0]: diff[0] + target_size[0], diff[1]: diff[1] + target_size[1],
               diff[2]: diff[2] + target_size[2]]

    @staticmethod
    def add_padding(layer, diff):
        return F.pad(layer, [diff[-1], diff[-1] - diff[-1] // 2, diff[-2] // 2, diff[-2] - diff[-2] // 2,
                             diff[-3] // 2, diff[-3] - diff[-3] // 2])

    # [N, C, Z, Y, X]; N - number of batches, C - number of channels
    # Two options for concatenation - crop x2 or add padding to x1
    def forward(self, x1, x2, concat='crop'):
        x1 = self.deconv(x1)
        layer_size = x1.size()[-3]
        target_size = x2.size()[-3]
        diff = (layer_size - target_size)

        if concat == 'crop':
            x2 = self.crop_centre(x2, target_size, diff)
        else:
            x1 = self.add_padding(x1, diff)

        x = torch.cat([x2, x1], dim=2)
        x = self.double_conv(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.down1 = Down(1, start_neurons * 1, do=0.25)
        self.down2 = Down(start_neurons * 1, start_neurons * 2)
        self.down3 = Down(start_neurons * 2, start_neurons * 4)
        self.down4 = Down(start_neurons * 4, start_neurons * 8)
        self.middle = nn.Sequential(
            nn.Conv3d(start_neurons * 8, start_neurons * 16, 3),
            nn.Conv3d(start_neurons * 16, start_neurons * 16, 3),
        )
        self.up4 = Up(start_neurons * 16, start_neurons * 8)
        self.up3 = Up(start_neurons * 8, start_neurons * 4)
        self.up2 = Up(start_neurons * 4, start_neurons * 2)
        self.up1 = Up(start_neurons * 2, start_neurons * 1)
        self.final = nn.Sequential(
            nn.Dropout3d(0.5),
            nn.Conv3d(start_neurons * 1, 1, 1)
        )

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x = self.middle(x4)
        x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        return self.final(x)


# # dice = 2*intersection/union
# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred = K.cast(y_pred, 'float32')
#     y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
#     intersection = y_true_f * y_pred_f
#     score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
#     return score
#
#
# def dice_loss(y_true, y_pred):
#     smooth = 1.
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = y_true_f * y_pred_f
#     score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#     return 1. - score
#
#
# # Index de Jaccard = intersection/union; distance de Jaccard = 1-index
# def jaccard_distance_loss(y_true, y_pred):
#     smooth = 100.
#     intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
#     sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
#     jac = (intersection + smooth) / (sum_ - intersection + smooth)
#     return (1 - jac) * smooth
#
#
# def bce_jaccard_loss(y_true, y_pred):
#     return binary_crossentropy(y_true, y_pred) + jaccard_distance_loss(y_true, y_pred)


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


# load image (DICOM -> numpy)
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

# each element of train_y represents 1 cube of 256 * 256 * 136 + metadata
train_y = [load_itk(label)[0] for label in label_files]

# Complement y with zeros
# train_y = np.array(
#     [np.concatenate((i, np.zeros((i.shape[0], i.shape[1], train_x.shape[2] - i.shape[2]))), axis=2) for i in train_y])
train_y = np.array(train_y)
train_x = np.array([train_x])

train_x = train_x[:, :, :, 58:186]
train_y = train_y[:, :, :, 4:132]

# Creates a fifth fictitious dimension for the number of channels (constraint of keras and pytorch)
# For pytorch the requested size is [N, C, Z, Y, X], for keras - [N, X, Y, Z, C]
# keras
# enter_shape = train_x.shape + (1,)
# train_x = train_x.reshape(*enter_shape)
# train_y = train_y.reshape(*enter_shape)
# pytorch
enter_shape = (train_x.shape[0], 1) + train_x.shape[-3:][::-1]
train_x = train_x.reshape(*enter_shape)
train_y = train_y.reshape(*enter_shape)

# Visualisation
# train_y = np.array(train_y[0])
# print(train_x.shape)
# mlab.contour3d(train_y)
# mlab.savefig('surface.obj')
# input()
#
tensor_x = torch.stack([torch.Tensor(i) for i in train_x])
tensor_y = torch.stack([torch.Tensor(i) for i in train_y])

train_dataset = data.TensorDataset(tensor_x, tensor_y)
train_loader = data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True)

test_dataset = data.TensorDataset(tensor_x, tensor_y)
test_loader = data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True)

net = Net()
optimizer = optim.SGD(net.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
print(net)

total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Passage through the network
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculation of precision
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))
