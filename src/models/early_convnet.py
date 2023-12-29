import keras

class EarlyConvnet(keras.Model):
    # TODO: implement EarlyConvent model (current state is as per: https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing#the_model_class)

    def __init__(self, num_classes=1000):
        super().__init__()
        self.block_1 = ResNetBlock()
        self.block_2 = ResNetBlock()
        self.global_pool = layers.GlobalAveragePooling2D()
        self.classifier = Dense(num_classes)

    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.global_pool(x)
        return self.classifier(x)