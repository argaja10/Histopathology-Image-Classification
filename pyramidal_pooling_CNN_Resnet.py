from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, Add, AveragePooling2D, Concatenate, Lambda
import tensorflow as tf





def residual_block(x, filters, kernel_size=3, stride=1, activation="relu", chanDim=-1):
    """Function to create a residual block with two convolutional layers."""
    # Save the input tensor
    shortcut = x
    
    # First convolutional layer
    x = Conv2D(filters, (kernel_size, kernel_size), strides=stride, padding="same")(x)
    x = Activation(activation)(x)
    x = BatchNormalization(axis=chanDim)(x)
    
    # Second convolutional layer
    x = Conv2D(filters, (kernel_size, kernel_size), padding="same")(x)
    x = BatchNormalization(axis=chanDim)(x)
    
    # Add a convolution to the shortcut if the number of filters changes
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding="same")(shortcut)
        shortcut = BatchNormalization(axis=chanDim)(shortcut)
    
    # Add shortcut to the output of the convolutional block
    x = Add()([x, shortcut])
    x = Activation(activation)(x)
    
    return x


def spatial_pyramid_pooling(x, pool_list, chanDim):
    """Function to create a spatial pyramid pooling module."""
    pooled_outputs = []
    height, width = x.shape[1:3]
    
    for pool_size in pool_list:
        if height >= pool_size and width >= pool_size:
            # Apply average pooling with the current pool size
            pooled = AveragePooling2D(pool_size=(pool_size, pool_size), strides=(pool_size, pool_size))(x)
            
            # Flatten the pooled output and add it to the list
            pooled = Flatten()(pooled)
            pooled_outputs.append(pooled)

    # Concatenate the pooled outputs along the channels
    x = Concatenate(axis=chanDim)(pooled_outputs)

    return x


class BC_Model:
    @staticmethod
    def build(width, height, depth, classes):
    
        inputShape = (height, width, depth)
        chanDim = -1
        
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        
        inputs = Input(shape=inputShape)
        
        x = Conv2D(32, (3, 3), padding="same")(inputs)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        
        # Adding the first residual block
        x = residual_block(x, 64, chanDim=chanDim)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        
        # Adding the second residual block
        x = residual_block(x, 128, chanDim=chanDim)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        
        # Adding the third residual block
        x = residual_block(x, 256, chanDim=chanDim)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        
        # Adding Spatial Pyramid Pooling Module
        x = spatial_pyramid_pooling(x, [1, 2, 4], chanDim=chanDim)
        
        x = Dense(512)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        outputs = Dense(classes)(x)
        outputs = Activation("softmax")(outputs)
        
        model = Model(inputs, outputs)
        
        return model
