import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk

data_path = '/home/203112001/MICCAI_BraTS_2019_Data_Training/'  # Replace this with the path to your dataset


train_images_array = np.load('train_images_array.npy')
train_masks_array = np.load('train_masks_array.npy')
train_images_T1 = np.load('train_images_T1.npy')
train_images_T1ce = np.load('train_images_T1ce.npy')
train_images_T2 = np.load('train_images_T2.npy')
train_images_FLAIR = np.load('train_images_FLAIR.npy')

train_masks_array = np.transpose(train_masks_array, (0, 2, 3, 4, 1))

import tensorflow as tf
from keras import backend as K
from sklearn.metrics import make_scorer
from tensorflow.keras.optimizers import Adam, RMSprop


def mean_io_u(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    mean_iou = K.mean((intersection + K.epsilon()) / (union + K.epsilon()))
    return mean_iou

# Register the custom metric function
tf.keras.utils.get_custom_objects()['mean_io_u'] = mean_io_u

def mean_io_u(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    mean_iou = K.mean((intersection + K.epsilon()) / (union + K.epsilon()))
    return mean_iou

# Register the custom metric function
tf.keras.utils.get_custom_objects()['mean_io_u'] = mean_io_u


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

# Register the custom metric function
tf.keras.utils.get_custom_objects()['dice_coef'] = dice_coef
# Create the scoring function using make_scorer
dice_scoring = make_scorer(dice_coef, greater_is_better=True)
# Create the scoring function using make_scorer
mean_scoring = make_scorer(mean_io_u, greater_is_better=True)

import os
import numpy as np
import nibabel as nib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Dropout, UpSampling3D, Concatenate, AveragePooling3D, Reshape, Dense, Multiply, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam


# Define the input image size (3D)
input_shape = (128, 128, 128, 4)  # Update the channels (4) based on the number of modalities (e.g., T1, T1ce, T2, FLAIR)

def se_block(input_tensor, ratio=4):

    """Squeeze-and-Excitation block implementation"""

    channels = input_tensor.shape[-1]
    x = K.mean(input_tensor, axis=0, keepdims=True)
    x = Conv3D(channels // ratio, 1, activation='relu', kernel_initializer='he_normal', use_bias=False)(x)
    x = Conv3D(channels, 1, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(x)
    return Multiply()([input_tensor, x])

def create_unet_model(input_shape=(128,128,128,4),
                          dropout_rate=0.2, learning_rate=0.001):
    print("inside unet")
    inputs = Input(shape=input_shape)
    print("inputststement run")
#-----------------------------------delete this part later-------------------------------
# Encoder

# Define the input shape (depth, height, width, modality)
#    depth = 128
#    height = 128
#    channels = 1 
#    width = 128
#    num_modalities = 4  # Update with your actual data dimensions
#    input_shapes = [(None, depth, height, width, channels) for _ in range(num_modalities)]
# Create a list to hold the separate modality input tensors
#    modality_inputs = [tf.keras.layers.Input(shape=shape) for shape in input_shapes]
#    print("code run till modality inputs statement")

# Create a list to hold the separate modality pathways

#    modality_pathways = []



# Iterate over each modality input and create a Conv3D pathway

#    for modality_input in modality_inputs:
#       print("inside modlity input loop")
#       modality_conv = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu')(modality_input)
#       print("xxmodality_convxxinside modlity input loop")
      # Add more Conv3D layers or other operations as needed for each modality
#       modality_pathways.append(modality_conv)
#       print("modality_pathway")

#    print("shape of modality_conv", modality_conv)
#    # Concatenate the outputs of the modality pathways
#    combined = tf.keras.layers.Concatenate()(modality_pathways)
#    print("Combined shape:", combined.shape)
    conv1 = BatchNormalization()(inputs)
    conv1 = Activation('relu')(conv1)
    print("Conv1 shape:", conv1.shape)
    conv1 = se_block(conv1)  # Add SE block
    print("After SE block, Conv1 shape:", conv1.shape)
    conv1 = Conv3D(64, 3, activation=None, padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = se_block(conv1)  # Add SE block
    print("shape of conv1:",conv1.shape)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv1)
#    pool1 = tf.keras.layers.Reshape(target_shape=(-1,128,128,128,64))(pool1)
    print("shape of pool1",pool1.shape)
     
    conv2 = Conv3D(128, 3, activation=None, padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    print("Conv2 shape:", conv2.shape)
    conv2 = se_block(conv2)  # Add SE block
    conv2 = Conv3D(128, 3, activation=None, padding='same')(conv2)
    print("shape of pool1:",pool1.shape)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = se_block(conv2)  # Add SE block
    print("shape of conv2:",conv2.shape)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv2)
    
#    pool2 = tf.keras.layers.Reshape(target_shape=(-1,64,64,64,64))(pool2)

    conv3 = Conv3D(256, 3, activation=None, padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = se_block(conv3)  # Add SE block
    conv3 = Conv3D(256, 3, activation=None, padding='same')(conv3)
    print("shape of pool2:",pool2.shape)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = se_block(conv3)  # Add SE block
    print("shape of conv3:",conv3.shape)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv3)
#    pool3 = tf.keras.layers.Reshape(target_shape=(-1,32,32,32,64))(pool3)

    conv4 = Conv3D(512, 3, activation=None, padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = se_block(conv4)  # Add SE block
    conv4 = Conv3D(512, 3, activation=None, padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = se_block(conv4)  # Add SE block
    drop4 = Dropout(dropout_rate)(conv4)
    print("shape of conv4:",conv4.shape)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv4)
#    pool4 = tf.keras.layers.Reshape(target_shape=(-1,16,16,16,64))(pool4)

    # ... (Rest of the encoder layers for 3D data)
    print("shape of drop4:",drop4.shape)
##    cpool4 = (UpSampling3D(size=(2, 2, 2))(pool4[...,0]))
#    cpool4 = tf.keras.layers.Reshape(target_shape=(-1,32,32,32,64))(cpool4)
    up5 = Conv3D(512, 3, activation=None, padding='same')(UpSampling3D(size=(2, 2, 2))(pool4))
    up5 = BatchNormalization()(up5)
    up5 = Activation('relu')(up5)
    print("shape of up5:",up5.shape)
    merge5 = Concatenate()([drop4, up5])
    conv5 = Conv3D(512, 3, activation=None, padding='same')(merge5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = se_block(conv5) # Add SE block
    conv5 = Conv3D(512, 3, activation=None, padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = se_block(conv5) # Add SE block
    print("shape of up5:",up5.shape)
    
#    cpool5 = (UpSampling3D(size=(2, 2, 2))(conv5[..., 0]))
#    cpool5 = tf.keras.layers.Reshape(target_shape=(-1,64,64,64,64))(cpool5)

    up6 = Conv3D(256, 3, activation=None, padding='same')(UpSampling3D(size=(2, 2, 2))(conv5))
    up6 = BatchNormalization()(up6)
    up6 = Activation('relu')(up6)
    merge6 = Concatenate()([conv3, up6])
    conv6 = Conv3D(256, 3, activation=None, padding='same')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = se_block(conv6)  # Add SE block
    conv6 = Conv3D(256, 3, activation=None, padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = se_block(conv6)  # Add SE block
    
#    cpool6 = (UpSampling3D(size=(2, 2, 2))(conv6[..., 0]))
#    cpool6 = tf.keras.layers.Reshape(target_shape=(-1,128,128,128,64))(cpool6)
    
    up7 = Conv3D(128, 3, activation=None, padding='same')(UpSampling3D(size=(2, 2, 2))(conv6))
    up7 = BatchNormalization()(up7)
    up7 = Activation('relu')(up7)
    merge7 = Concatenate()([conv2, up7])
    conv7 = Conv3D(128, 3, activation=None, padding='same')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = se_block(conv7)  # Add SE block
    conv7 = Conv3D(128, 3, activation=None, padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = se_block(conv7)  # Add SE block
    print("shape of Conv2:",conv2.shape)
    print("shape of up7:",up7.shape)

#    cpool7 = (UpSampling3D(size=(2, 2, 2))(conv7[..., 0]))
#    cpool7 = tf.keras.layers.Reshape(target_shape=(-1,128,128,128,64))(cpool7)

    up8 = Conv3D(64, 3, activation=None, padding='same')(UpSampling3D(size=(2, 2, 2))(conv7))
    up8 = BatchNormalization()(up8)
    up8 = Activation('relu')(up8)
    print("shape of up8:",up8.shape)
    merge8 = Concatenate()([conv1, up8])
    conv8 = Conv3D(64, 3, activation=None, padding='same')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = se_block(conv8)  # Add SE block
    conv8 = Conv3D(64, 3, activation=None, padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = se_block(conv8)  # Add SE block

    # ... (Rest of the decoder layers for 3D data)

    # Output
    conv9 = Conv3D(1, 1, 1, activation='sigmoid')(conv8)  # Use 3D Conv3D and adjust the number of output channels based on your task
    model = Model(inputs=inputs, outputs=conv9)
    model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy','mean_io_u', 'dice_coef'])
    return model

print("Code has run till start of model.create")
input_shape = (128, 128, 128, 4)
model=create_unet_model(input_shape=input_shape)
model.summary()
print("code has completed the create part")
batch_size = 16
epochs =1000


print("code has started the fit part")

# Create a list of input tensors for each modality
input_tensors = [train_images_T1, train_images_T1ce, train_images_T2, train_images_FLAIR]
input_tensors = np.concatenate([train_images_T1, train_images_T1ce, train_images_T2, train_images_FLAIR], axis=-1)

from tensorflow.keras.callbacks import Callback, EarlyStopping
class MultiMetricEarlyStopping(Callback):
    def __init__(self, monitor_metrics, target_values, patience=0, restore_best_weights=True):
        super(MultiMetricEarlyStopping, self).__init__()
        self.monitor_metrics = monitor_metrics
        self.target_values = target_values
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best = np.Inf  # Initialize with a high value

    def on_epoch_end(self, epoch, logs=None):
        current_metrics = [logs.get(metric) for metric in self.monitor_metrics]
        all_metrics_above_target = all(metric >= target for metric, target in zip(current_metrics, self.target_values))

        if all_metrics_above_target:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.best_weights is not None:
                        self.model.set_weights(self.best_weights)
        else:
            self.wait = 0
            if all(metric < target for metric, target in zip(current_metrics, self.target_values)):
                self.best = np.Inf
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"Early stopping after epoch {self.stopped_epoch}.")

# Define the metrics you want to monitor
monitor_metrics = ['mean_io_u', 'dice_coef']

# Define the target values for each metric
target_values = [0.85, 0.85]  # Adjust these values to your desired targets

# Create the custom callback
multi_metric_early_stopping = MultiMetricEarlyStopping(
    monitor_metrics=monitor_metrics,
    target_values=target_values,
    patience=10,
    restore_best_weights=True
)

callbacks = [multi_metric_early_stopping]
# Pass the list of input tensors to the model.fit function
history = model.fit(
    input_tensors,
    train_masks_array,  # Provide the corresponding masks
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks
)
print("fit part executed")
import pickle

# Save the history object to a file
with open('/home/203112001/history3d500epoch.pickle', 'wb') as f:
    pickle.dump(history, f)

