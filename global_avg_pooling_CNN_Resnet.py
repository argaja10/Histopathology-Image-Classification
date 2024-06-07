
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, Add, GlobalAveragePooling2D


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
    
    # Add shortcut to the output of the convolutional block
    x = Add()([x, shortcut])
    x = Activation(activation)(x)
    
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
        
        #x = Flatten()(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(1)(x)
        #x = Activation("relu")(x)
        #x = BatchNormalization()(x)
        #x = Dropout(0.5)(x)
        
        #outputs = Dense(classes)(x)
        outputs = Activation("sigmoid")(x)
        
        model = Model(inputs, outputs)
        
        return model
