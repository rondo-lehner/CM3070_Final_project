import tensorflow as tf

class FCN_32s(tf.keras.Model):

    def __init__(self, num_classes=7, img_size=224):
        super(FCN_32s, self).__init__()

        # VGG network without the fully connected layers
        # Note: each Keras Application expects a specific kind of input preprocessing. For VGG16, call tf.keras.applications.vgg16.preprocess_input on your inputs before passing them to the model. vgg16.preprocess_input will convert the input images from RGB to BGR, then will zero-center each color channel with respect to the ImageNet dataset, without scaling.
        self.base_model = tf.keras.applications.VGG16(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=(img_size, img_size, 3),
        )

        # Layers that need to be "fully convolutionalized" (layer sizes as per https://doi.org/10.48550/arXiv.1409.1556, last accessed: 22 Jan 2024)
        # Conversion as per: https://cs231n.github.io/convolutional-networks/#convert, last accessed: 22 Jan 2024   
        
        ## FC-4096
        self.block_6_conv1 = tf.keras.layers.Conv2D(
            filters=4096,
            kernel_size=(7,7),
            strides=(1,1),
            padding='same',
            activation='relu',
            name='block_6_conv1'
        )
        ## FC-4096
        self.block_6_conv2 = tf.keras.layers.Conv2D(
            filters=4096,
            kernel_size=(7,7),
            strides=(1,1),
            padding='same',
            activation='relu',
            name='block_6_conv2'
        )
        ## FC-1000 --> This final classifier layer is discarded
        # DO NOTHING

        ## We append a 1 × 1 convolution with channel dimension 21 to predict scores for each of the PASCAL classes at each of the coarse output locations
        # Looking at the original implementation there is no explicit activation function: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn32s/net.py, last accessed 22.01.2024
        # Hence "linear" is chosen
        self.block_6_conv3 = tf.keras.layers.Conv2D(
            filters=7,
            kernel_size=(1,1),
            strides=(1,1),
            padding='same',
            activation='linear',
            name='block_6_conv3_score'
        )
        ## followed by a deconvolution layer to bi-linearly upsample the coarse outputs to pixel-dense outputs
        self.block_6_deconv1 = tf.keras.layers.Conv2DTranspose(
            filters=7,
            kernel_size=(64, 64),
            strides=(32, 32),
            use_bias=False, # As per original implementation
            padding='same',
            activation='softmax',
            name='block_6_deconv1'
        )
        self.block_6_deconv1.trainable = False # As per original implementation: param=[dict(lr_mult=0)]


    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.block_6_conv1(x)
        x = self.block_6_conv2(x)
        x = self.block_6_conv3(x)
        return self.block_6_deconv1(x)

    def build_graph(self, raw_shape):
        x = tf.keras.Input(shape=raw_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))