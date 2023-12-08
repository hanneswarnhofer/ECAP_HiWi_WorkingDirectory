from tools.utils import to_list, SearchAttribute
import inspect
from models.base_metrics import tf_not_implemented, torch_not_implemented
from data.data import Array


class Task():
    """ Base class for defining supervised learning tasks
        Sets losses for TF/Keras and torch models.
        Manages the metrics for evaluation and live monitoring during training.
    """

    def __init__(self, label, unit):
        self.tf_loss_fn = tf_not_implemented
        self.torch_loss_fn = torch_not_implemented
        self.label = label
        self.unit = unit
        self.name = label.name

    def loss(self, dtype):
        if dtype == "tf":
            loss_fn = self.tf_loss_fn
        elif dtype == "torch":
            loss_fn = self.torch_loss_fn
        else:
            raise TypeError("training has to be of type 'torch' or 'tf'.")
        return loss_fn() if inspect.isclass(loss_fn) is True else loss_fn

    def set_loss(self, loss_fn, tf_name=None, torch_name=None):

        if type(loss_fn) == str:
            tf_name = loss_fn if tf_name is None else tf_name
            torch_name = loss_fn if torch_name is None else torch_name

            if "Loss" not in torch_name:
                torch_name = torch_name + "Loss"

            search_tf = SearchAttribute(tf_name)
            search_tf.search("from tensorflow.keras import losses as tf_losses")
            search_tf.search("from models.tf import tf_metrics", not_found_error=False)

            loss = search_tf.end()

            if loss is not None:
                self.tf_loss_fn = loss

            search_torch = SearchAttribute(torch_name)
            search_torch.search("from torch.nn.modules import loss as torch_losses")
            search_torch.search("from models.torch import torch_metrics", not_found_error=False)

            loss = search_torch.end()

            if loss is not None:
                self.torch_loss_fn = loss

        elif callable(loss_fn) is True:  # if functionen / class given
            if "keras" or "tensorflow" in loss_fn.__module__:
                self.tf_loss_fn = loss_fn
            elif "torch" in loss_fn.__module__:
                self.torch_loss_fn = loss_fn
            else:
                raise NameError("loss_fn %s seems not to be tf/keras and torch or torch function" % loss_fn.__name__)
        else:
            raise TypeError("loss_fn has to be of type 'str' or fn.")

    def add_metric(self, metric):
        self.metrics += to_list(metric)

    @property
    def __name__(self):
        return self.__class__.__name__


class Classification(Task):
    """
        Class for setting up a (supervised) classification task using Keras/TF or torch/PyG model.



    Parameters
    ----------
    num_classes : int
        Number of classes/targets.
    classes : list of str
        Names of the classes
    loss_fn : keras.fn / torch.fn
        Loss function that should be used for training (Default: Crossentropy).

    Attributes
    ----------
    metrics : type
        Description of attribute `metrics`.


    """

    def __init__(self, label, classes=None, loss_fn="Crossentropy", metrics="default", unit=None):
        super().__init__(label, unit)
        assert isinstance(label, Array), "Given label object has to be of type 'astro_dl.data.data.Array'."
        assert label().ndim > 1, "Detected shape: %s. Given label set has to be one-hot encoded" % (label().shape,)
        num_classes = label().shape[-1]
        self.num_classes = num_classes
        self.classes = classes
        self.nodes = self.num_classes
        self.metrics = []
        self.normalize = "softmax"

        if classes is not None:
            assert len(classes) == len(self.num_classes), "classes (class labels) have to be of same length as num_classes"

        if loss_fn == "Crossentropy":
            if num_classes == 1:
                self.set_loss("BinaryCrossentropy", torch_name="BinaryCrossEntropy")
            elif num_classes > 1:
                self.set_loss("CategoricalCrossentropy")
        else:
            self.set_loss(loss_fn)

        if metrics == "default":
            from models.metrics import auroc_fn, accuracy_fn
            from models.base_metrics import ROCMetric, ConfusionMetric, ScoreMetric

            if num_classes == 2:
                auroc = ROCMetric(auroc_fn, num_classes)
                self.metrics.append(auroc)

            confusion = ConfusionMetric(accuracy_fn, num_classes)
            self.metrics.append(confusion)

            score = ScoreMetric(accuracy_fn, num_classes, name="score")
            self.metrics.append(score)
        else:
            self.metrics = to_list(metrics)

    @property
    def metric_dict(self):
        return {met.name: met for met in self.metrics}


class Regression(Task):
    def __init__(self, label, loss_fn='MSE', metrics="default", normalize="default", unit=None):
        super().__init__(label, unit)
        assert isinstance(label, Array), "Given label object has to be of type 'astro_dl.data.data.Array'."
        assert label().ndim == 2, "Detected shape: %s. Given label set has to have dummy axis, i.e., shape=(n, 1)" % (label().shape,)
        assert label().shape[-1] == 1, "Detected shape: %s. Given label set has to be scalar shape=(n, 1). If your label is a vector shape=(n, m)), use Regression() instead of Vectorregression()" % label().shape
        assert normalize in ["default", "standard", "euclidean", "min_max", "softmax"], "Normalization has to be 'standard', 'euclidean', or 'min_max'."

        if normalize == "default":
            normalize = "standard"

        self.normalize = normalize
        self.nodes = label().shape[-1]
        self.set_loss(loss_fn)

        if metrics == "default":
            from models.base_metrics import RegressionMetric, ScatterMetric
            from models.metrics import bias_fn, resolution_fn, correlation_fn, rel_resolution_fn, rel_bias_fn

            bias = RegressionMetric(bias_fn, unit=self.unit)
            rel_bias = RegressionMetric(rel_bias_fn, unit=self.unit)
            resolution = RegressionMetric(resolution_fn, unit=self.unit)
            rel_resolution = RegressionMetric(rel_resolution_fn)
            correlation = ScatterMetric(correlation_fn, unit=self.unit)
            self.metrics = [bias, rel_bias, resolution, rel_resolution, correlation]
        else:
            self.metrics = to_list(metrics)


class VectorRegression(Task):
    """
        Type for more sophisticated regression tasks. Set meaningful defaults for training metrics. MSE is used as loss (default). For example, to reconstruct arrival direction the shower axis should be used as target to circumvent a pole at theta=0.
        Currently, the class supports to types:
        - angular: for reconstructing angular directions (shower axis)
        - distance: for reconstructing positions (e.g., impact point)
    """

    def __init__(self, label, vec_type, loss_fn='MSE', metrics="default", unit=None, normalize="default"):
        super().__init__(label, unit)
        self.set_loss(loss_fn)
        assert isinstance(label, Array), "Given label object has to be of type 'astro_dl.data.data.Array'."
        assert label().ndim == 2, "Detected shape: %s. Given label for Vector regression has to be a vector." % (label().shape,)
        assert label().shape[-1] > 1, "Detected shape: %s. If the shape of your label is 1, use Regression() instead of Vectorregression()" % label().shape

        self.nodes = label.shape[-1]
        assert normalize in ["default", "standard", "euclidean", "min_max", "softmax"], "Normalization has to be 'standard', 'euclidean', or 'min_max'."
        self.normalize = normalize

        if metrics == "default":
            from models.base_metrics import DistanceMetric

            if vec_type == "angular":
                from models.metrics import angular_resolution_fn
                from plotting.utils import calc_angulardistance
                angular_resolution = DistanceMetric(angular_resolution_fn, agg_fn=calc_angulardistance, unit=self.unit)

                if normalize == "default":
                    self.normalize = "euclidean"

                self.metrics = [angular_resolution]
            elif vec_type == "distance":
                from models.metrics import euclidean_resolution_fn
                from plotting.utils import calc_distance

                euclidean_resolution = DistanceMetric(euclidean_resolution_fn, agg_fn=calc_distance, unit=self.unit)
                self.metrics = [euclidean_resolution]
            else:
                raise TypeError("Vector regression has to be of vec_type 'angular' or 'distance'")
        else:
            if vec_type(metrics) == str:
                self.metrics = []
                print("No metric input to VectorRegression")
            else:
                self.metrics = to_list(metrics)

        # HERE ESTIMATE ANGULARDISTANCE / or SPATIAL DISTANCE BEFORE APLLYING METRIC
        # ODER komplett eigene Metrics fuer VectorRegression <-- besser?!
