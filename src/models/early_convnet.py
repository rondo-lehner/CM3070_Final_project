import tensorflow as tf
import src.layers.subsampling as subsampling

class EarlyConvnet(tf.keras.Model): # TODO: change to tf.keras.Model
    # TODO: implement EarlyConvent model (current state is as per: https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing#the_model_class)

    def __init__(self, num_classes=7):
        super(EarlyConvnet, self).__init__()
        ## Layers:
        # I0 - Input: assuming 40 x 40 x 3
        ## C1 - 7x7 convolution kernel (6 Feature maps @ 34 x 34)
        self.C1 = tf.keras.layers.Conv2D(
            name = 'C1',
            filters = 6,
            kernel_size = (7, 7),
            use_bias = True,                # Network formula shows b_{ij}, implying biases for each feature map (j) and layer (i)
            strides = (1, 1),
            padding = 'valid',
            data_format = 'channels_last'
        )

        self.A1 = tf.keras.layers.Activation('sigmoid')

        ## S2 - 2x2 subsampling layer (6 Feature maps @ 17 x 17)
        self.S2 = subsampling.Subsampling(
            name = 'S2',
            pool_size = (2, 2),
            strides = (2, 2),
            padding = 'valid'
        )

        self.A2 = tf.keras.layers.Activation('tanh')

        ## C3 - 6x6 conv kernel (16 FM @ 12 x 12)
        self.C3 = tf.keras.layers.Conv2D(
            name = 'C3',
            filters = 16,
            kernel_size = (6, 6),
            use_bias = True,                # Network formula shows b_{ij}, implying biases for each feature map (j) and layer (i)
            strides = (1, 1),
            padding = 'valid',
            data_format = 'channels_last'
        )

        self.A3 = tf.keras.layers.Activation('sigmoid')

        ## S4 - 2x2 subsampling layer (16 FM @ 6x6)
        self.S4 = subsampling.Subsampling(
            name = 'S4',
            pool_size = (2, 2),
            strides = (2, 2),
            padding = 'valid'
        )
        
        self.A4 = tf.keras.layers.Activation('tanh')
        
        # C5 - 6x6 conv kernel (40 FM @ 1x1)
        self.C5 = tf.keras.layers.Conv2D(
            name = 'C5',
            filters = 40,
            kernel_size = (6, 6),
            use_bias = True,                # Network formula shows b_{ij}, implying biases for each feature map (j) and layer (i)
            strides = (1, 1),
            padding = 'valid',
            data_format = 'channels_last'
        )

        self.A5 = tf.keras.layers.Activation('sigmoid')

        # F6 - Final output layer (N @ 1x1) N number of classes
        self.F6 = tf.keras.layers.Dense(
            name = 'F6',
            units = num_classes       
        )

        self.A6 = tf.keras.layers.Softmax() # No activation function specified by the paper, my assumption is that softmax is a reasonable choice

    def call(self, inputs):
        x = self.C1(inputs)
        x = self.A1(x)
        x = self.S2(x)
        x = self.A2(x)
        x = self.C3(x)
        x = self.A3(x)
        x = self.S4(x)
        x = self.A4(x)
        x = self.C5(x)
        x = self.A5(x)
        x = self.F6(x)
        return self.A6(x)

    def build_graph(self, raw_shape):
        x = tf.keras.Input(shape=raw_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))