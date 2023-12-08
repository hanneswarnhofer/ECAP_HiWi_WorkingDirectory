from tensorflow.keras import backend as K
from tensorflow import keras
lay = keras.layers


class EuclideanNorm(lay.Layer):
    """ Custom layer: Normalizes 3-vectors to 1 (euclidean norm) """

    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(EuclideanNorm, self).__init__(**kwargs)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(EuclideanNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        super(EuclideanNorm, self).build(input_shape)

    def call(self, x):
        return K.l2_normalize(x, axis=self.axis)


def NormToLabel(stats, factor=1, name=None):
    """ Custom layer to circumvent normalization of physical outputs. """
    return lay.Lambda(lambda x: factor * stats['std'] * x + stats['mean'], name=name)
