import tensorflow as tf

@tf.function
def downsample_block(x, n_filters, n_filters_prev):
    # drawing the initial weights from a Gaussian distribution with a standard deviation of p2/N, where N denotes the number of incoming nodes of one neu- ron
    N = 9 * n_filters_prev

    x = tf.keras.layers.Conv2D(
        filters=n_filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='valid',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation='relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=tf.math.sqrt(2/N))
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=n_filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='valid',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation='relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=tf.math.sqrt(2/N))
    )(x)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        data_format='channels_last'
    )(x)

    return x

def getUNet(n_downsampling=4, input_shape=(224, 244, 3)):

    inputs = tf.keras.Input(shape=input_shape)

    """
    DOWNSAMPLING
    The contracting path follows the typical architecture of a convolutional network.
    It consists of the repeated application of two 3x3 convolutions (unpadded convolutions),
    each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with
    stride 2 for downsampling. At each downsampling step we double the number of feature channels.
    """
    filter_sizes = [64 ** i for i in range(n_downsampling)]
    for i, filter_size in enumerate(filter_sizes):
        if i == 0:
            x = downsample_block(inputs, filter_size, 3)
        else:
            x = downsample_block(x, filter_size, filter_sizes[i-1])

    return tf.keras.Model(inputs=inputs, outputs=x)

    


    """
    UPSAMPLING
    Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2
    convolution (“up-convolution”) that halves the number of feature channels, a concatenation with
    the correspondingly cropped feature map from the contracting path, and two 3x3 convolutions,
    each followed by a ReLU.
    """



def upsample_block(input_size):
    pass