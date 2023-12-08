import torchmetrics as tm
import torch


class TorchMetric():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self._get_name()


class Accuracy(TorchMetric, tm.Accuracy):
    def __call__(self, preds, targets):
        return super().__call__(preds, targets.int())


class CategoricalCrossentropyLoss(TorchMetric, torch.nn.NLLLoss):
    def __call__(self, input, target):
        return super().__call__(torch.clamp(torch.log(input + 1E-8), -100, 0), torch.argmax(target, axis=-1))


class Auroc(TorchMetric, tm.AUROC):
    def __call__(self, preds, targets):
        return super().__call__(preds, targets.int())


class Correlation(TorchMetric, tm.PearsonCorrCoef):
    def __call__(self, preds, targets):
        return super().__call__(preds, targets)


class Mean(TorchMetric, tm.MeanMetric):
    def __call__(self, preds, targets):
        return super().__call__(preds, targets.int())


# Aliases
Corr = Correlation
Acc = Accuracy
Bias = Mean
