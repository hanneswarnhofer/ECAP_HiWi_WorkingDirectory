from keras import backend as K
import tensorflow as tf
import numpy as np
from tensorflow.keras.metrics import AUC, CategoricalAccuracy  # , Metric as StatefulMetric


class TFmetric():
    def __init__(self):
        self.name = self.__class__.__name__.lower()
        self.__name__ = self.name.lower()


class Correlation(TFmetric):

    def __call__(self, y_true, y_pred):
        x = y_true
        y = y_pred
        mx = K.mean(x, axis=0)
        my = K.mean(y, axis=0)
        xm, ym = x - mx, y - my
        r_num = K.sum(xm * ym)
        x_square_sum = K.sum(xm * xm)
        y_square_sum = K.sum(ym * ym)
        r_den = K.sqrt(x_square_sum * y_square_sum)
        r = r_num / r_den
        return K.mean(r)


class Auroc(AUC):
    def __init__(self, num_classes, **kwargs):
        super().__init__(curve="ROC", **kwargs)
        self.num_classes = num_classes

#
# auroc = tf.keras.metrics.AUC(curve="ROC")


class Bias(TFmetric):

    def __call__(self, y_true, y_pred):
        mean, var = tf.nn.moments((y_true - y_pred), axes=[0])
        return mean


# class Euclidean_resolution(Metric):
#     pass

class Resolution(TFmetric):
    ''' Metric to control for standart deviation.
    To be implemented: Use Welford's algorithm for stateful estimation.'''

    # def __init__(self):
    #
    # def update(self, existingAggregate, newValue):
    #     (count, mean, M2) = existingAggregate
    #     count += 1
    #     delta = newValue - mean
    #     mean += delta / count
    #     delta2 = newValue - mean
    #     M2 += delta * delta2
    #     return (count, mean, M2)
    #
    # # Retrieve the mean, variance and sample variance from an aggregate
    # def finalize(self, existingAggregate):
    #     (count, mean, M2) = existingAggregate
    #     if count < 2:
    #         return float("nan")
    #     else:
    #         (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
    #         return (mean, variance, sampleVariance)

    def __call__(self, y_true, y_pred):
        mean, var = tf.nn.moments((y_true - y_pred), axes=[0])
        return tf.sqrt(var)


class Accuracy(CategoricalAccuracy):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes


class Rms(TFmetric):
    ''' Metric to control for Root Mean Squared Error '''

    def __call__(self, y_true, y_pred):
        return tf.sqrt(tf.reduce_mean((y_true - y_pred)**2))


class Angular_resolution(TFmetric):
    ''' Metric to control the mean of the angulardistance distribution '''

    def __call__(self, y_true, y_pred):
        return 180 / np.pi * tf.reduce_mean(tf.acos(tf.minimum(tf.reduce_sum(y_pred * y_true, axis=-1) / (tf.norm(y_pred, 'euclidean', axis=-1) * tf.norm(y_true, 'euclidean', axis=-1)), 1)))


class Merit(TFmetric):
    ''' Metric to control the mean of the angulardistance distribution '''

    def __call__(self, y_true, y_pred):
        return np.abs(y_true.mean() - y_pred.mean()) / np.sqrt(y_true.std()**2 + y_pred.std()**2)
        return np.abs(y_true.mean() - y_pred.mean()) / np.sqrt(y_true.std()**2 + y_pred.std()**2)
