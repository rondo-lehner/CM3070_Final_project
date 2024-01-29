import tensorflow as tf

# Implemented using Functional API instead of subclassing
def get_fcn_32s():
    # VGG network without the fully connected layers
    # Note: each Keras Application expects a specific kind of input preprocessing. For VGG16, call tf.keras.applications.vgg16.preprocess_input on your inputs before passing them to the model. vgg16.preprocess_input will convert the input images from RGB to BGR, then will zero-center each color channel with respect to the ImageNet dataset, without scaling.
    base_model = tf.keras.applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=(224, 224, 3), 
    )

    base_model.trainable = False

    # Layers that need to be "fully convolutionalized" (layer sizes as per https://doi.org/10.48550/arXiv.1409.1556, last accessed: 22 Jan 2024)
    # Conversion as per: https://cs231n.github.io/convolutional-networks/#convert, last accessed: 22 Jan 2024   
    
    ## FC-4096
    block_6_conv1 = tf.keras.layers.Conv2D(
        filters=4096,
        kernel_size=(7,7),
        strides=(1,1),
        padding='same',
        activation='relu',
        name='block_6_conv1'
    )
    ## FC-4096
    block_6_conv2 = tf.keras.layers.Conv2D(
        filters=4096,
        kernel_size=(1,1),
        strides=(1,1),
        padding='same',
        activation='relu',
        name='block_6_conv2'
    )
    ## FC-1000 --> This final classifier layer is discarded
    # DO NOTHING

    ## We append a 1 × 1 convolution with channel dimension 21 to predict scores for each of the PASCAL classes at each of the coarse output locations
    # Looking at the original implementation there is no explicit activation function: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn32s/net.py, last accessed 22.01.2024
    # I first opted for "linear" activation but then adapted the same activation as https://keras.io/examples/vision/fully_convolutional_network/ (last accessed 29.01.2024)
    # TODO: make 0-initialized as per Experimental framework of original paper
    block_6_conv3 = tf.keras.layers.Conv2D(
        filters=7,
        kernel_size=(1,1),
        strides=(1,1),
        padding='same',
        activation='relu',
        name='block_6_conv3_score'
    )
    block_6_softmax4 = tf.keras.layers.Conv2D(
        filters=7,
        kernel_size=(1, 1),
        activation='softmax',
        padding='same',
        strides=(1,1),
        name='block_6_softmax'
    )
    ## followed by a deconvolution layer to bi-linearly upsample the coarse outputs to pixel-dense outputs
    #  as per https://keras.io/examples/vision/fully_convolutional_network/ (last accessed 29.01.2024)
    block_6_deconv5 = tf.keras.layers.UpSampling2D(
        size=(32, 32),
        data_format='channels_last',
        interpolation='bilinear'
    )

    # base_model.trainable = False

    x = base_model.output
    x = block_6_conv1(x)
    x = block_6_conv2(x)
    x = block_6_conv3(x) #scoring layer
    x = block_6_softmax4(x)
    output_layer = block_6_deconv5(x)

    return tf.keras.Model(inputs=base_model.input, outputs=output_layer)