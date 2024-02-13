import tensorflow as tf

class ContractingBlock(tf.keras.Model):
    """
    DOWNSAMPLING
    The contracting path follows the typical architecture of a convolutional network.
    It consists of the repeated application of two 3x3 convolutions (unpadded convolutions),
    each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with
    stride 2 for downsampling. At each downsampling step we double the number of feature channels.
    """
    def __init__(self, n_filters, n_filters_prev):
        super(ContractingBlock, self).__init__()
        N1 = 9 * n_filters_prev
        N2 = 9 * n_filters      # second layer connects to less feature channels

        self.c1 = tf.keras.layers.Conv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            data_format='channels_last',
            dilation_rate=(1, 1),
            activation='relu',
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=tf.math.sqrt(2/N1))
        )

        self.c2 = tf.keras.layers.Conv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            data_format='channels_last',
            dilation_rate=(1, 1),
            activation='relu',
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=tf.math.sqrt(2/N2))
        )

        self.p = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid',
            data_format='channels_last'
        )

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.c2(x)
        output = self.p(x)
        return output

class ExpansiveBlock(tf.keras.Model):
    """
    UPSAMPLING
    Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2
    convolution (“up-convolution”) that halves the number of feature channels, a concatenation with
    the correspondingly cropped feature map from the contracting path, and two 3x3 convolutions,
    each followed by a ReLU.
    """
    def __init__(self, n_filters):
        super(ExpansiveBlock, self).__init__()
        N1 = 9 * n_filters * 2
        N2 = 9 * n_filters      # second layer connects to less feature channels
        
        self.u = tf.keras.layers.Conv2DTranspose(
            filters=n_filters,
            kernel_size=(2, 2),
            strides=(2, 2)
        )
    
        self.c1 = tf.keras.layers.Conv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            data_format='channels_last',
            dilation_rate=(1, 1),
            activation='relu',
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=tf.math.sqrt(2/N1))
        )

        self.c2 = tf.keras.layers.Conv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            data_format='channels_last',
            dilation_rate=(1, 1),
            activation='relu',
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=tf.math.sqrt(2/N2))
        )
        

    def call(self, inputs, inputs_copied):
        x = self.u(inputs)
        dim_target = tf.shape(x)[1]
        dim_source = tf.shape(inputs_copied)[1]
        delta = tf.math.subtract(dim_source, dim_target)
        offset = tf.math.floordiv(delta, tf.constant(2))
        inputs_cropped = tf.image.crop_to_bounding_box(
            image=inputs_copied,
            offset_height=offset,
            offset_width=offset,
            target_height=dim_target,
            target_width=dim_target
        )
        x = tf.concat([x, inputs_cropped], axis=-1)
        x = self.c1(x)
        output = self.c2(x)
        return output


def getUNet(n_downsampling=4, input_shape=(572, 572, 3)):

    inputs = tf.keras.Input(shape=input_shape)

    
    filter_sizes = [64 * ((i+1) ** 2) for i in range(n_downsampling)]

    blocks = []

    for i, filter_size in enumerate(filter_sizes):
        blocks.append(ContractingBlock(
            filter_size,
            (3 if i == 0 else filter_sizes[i-1])        # 3 channels for first block
        )(inputs if i == 0 else blocks[i-1]))           # connect layers

    # TODO: debug why expansive path does not reduce border pixels after convolutions
    for i, filter_size in enumerate(reversed(filter_sizes)):
        blocks.append(ExpansiveBlock(filter_size)(
            blocks[-1], # input from previous layer
            blocks[len(filter_sizes) - 2 - i]                         # skip connection
        ))


    return tf.keras.Model(inputs=inputs, outputs=blocks[-1])

    


    



def upsample_block(input_size):
    pass