import keras
import src.layers.subsampling

class EarlyConvnet(keras.Model):
    # TODO: implement EarlyConvent model (current state is as per: https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing#the_model_class)

    def __init__(self, num_classes=7):
        super().__init__()
        ## Layers:
        # I0 - Input: assuming 40 x 40 x 3
        ## C1 - 7x7 convolution kernel (6 Feature maps @ 34 x 34)
        self.C1 = keras.layers.Conv2D(
            filters = 6,
            kernel_size = (7, 7),
            activation = 'sigmoid',
            use_bias = True,                # Network formula shows b_{ij}, implying biases for each feature map (j) and layer (i)
            strides = (1, 1),
            padding = 'valid',
            data_format = 'channels_last',
        )

        ## S2 - 2x2 subsampling layer (6 Feature maps @ 17 x 17)
        self.S2 = Subsampling(
            pool_size = (2, 2),
            padding = 'valid',
            activation = tf.keras.activations.tanh
        )

        ## C3 - 6x6 conv kernel (16 FM @ 12 x 12)
        self.C3 = keras.layers.Conv2D(
            filters = 16,
            kernel_size = (6, 6),
            activation = 'sigmoid',
            use_bias = True,                # Network formula shows b_{ij}, implying biases for each feature map (j) and layer (i)
            strides = (1, 1),
            padding = 'valid',
            data_format = 'channels_last'
        )

        ## S4 - 2x2 subsampling layer (16 FM @ 6x6)
        self.S4 = Subsampling(
            pool_size = (2, 2),
            padding = 'valid',
            activation = tf.keras.activations.tanh
        )
        
        # C5 - 6x6 conv kernel (40 FM @ 1x1)
        self.C5 = keras.layers.Conv2D(
            filters = 40,
            kernel_size = (6, 6),
            activation = 'sigmoid',
            use_bias = True,                # Network formula shows b_{ij}, implying biases for each feature map (j) and layer (i)
            strides = (1, 1),
            padding = 'valid',
            data_format = 'channels_last'
        )

        # F6 - Final output layer (N @ 1x1) N number of classes
        self.F6 = keras.layers.Dense(
            units = num_classes,
            activation = 'softmax'          # No activation function specified by the paper, my assumption is that softmax is a reasonable choice
        )

    def call(self, inputs):
        x = self.C1(inputs)
        x = self.S2(x)
        x = self.C3(x)
        x = self.S4(x)
        x = self.C5(x)
        return self.F6(x)