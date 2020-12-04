import numpy as np
import torch

from .miou import compute_iu, fast_cm
from torchvision.utils import save_image
from random import *


class MeanIoU:
    """Mean-IoU computational block for semantic segmentation.

    Args:
      num_classes (int): number of classes to evaluate.

    Attributes:
      name (str): descriptor of the estimator.

    """

    def __init__(self, num_classes):
        if isinstance(num_classes, (list, tuple)):
            num_classes = num_classes[0]
        assert isinstance(
            num_classes, int
        ), f"Number of classes must be int, got {num_classes}"
        self.num_classes = num_classes
        self.name = "meaniou"
        self.reset()

    def reset(self):
        self.cm = np.zeros((self.num_classes, self.num_classes), dtype=int)

    def update(self, pred, gt):
        idx = gt < self.num_classes
        pred_dims = len(pred.shape)
        assert (pred_dims - 1) == len(
            gt.shape
        ), "Prediction tensor must have 1 more dimension that ground truth"
        if pred_dims == 3:
            class_axis = 0
        elif pred_dims == 4:
            class_axis = 1
        else:
            raise ValueError("{}-dimensional input is not supported".format(pred_dims))
        
        assert (
            pred.shape[class_axis] == self.num_classes
        ), "Dimension {} of prediction tensor must be equal to the number of classes".format(
            class_axis
        )
        pred = pred.argmax(axis=class_axis)

        # print("validate_test")
        # print(torch.from_numpy(pred).shape)
        # print(torch.from_numpy(gt).shape)
        # save_image(torch.from_numpy(pred).type(torch.FloatTensor) , "./Dataset2/test/out.png")
        # save_image(torch.from_numpy(gt).type(torch.FloatTensor) , "./Dataset2/test/target.png")

        self.cm += fast_cm(
            pred[idx].astype(np.uint8), gt[idx].astype(np.uint8), self.num_classes
        )

    def updateTest(self, pred, gt):
        idx = gt < self.num_classes
        pred_dims = len(pred.shape)
        assert (pred_dims - 1) == len(
            gt.shape
        ), "Prediction tensor must have 1 more dimension that ground truth"
        if pred_dims == 3:
            class_axis = 0
        elif pred_dims == 4:
            class_axis = 1
        else:
            raise ValueError("{}-dimensional input is not supported".format(pred_dims))
        
        assert (
            pred.shape[class_axis] == self.num_classes
        ), "Dimension {} of prediction tensor must be equal to the number of classes".format(
            class_axis
        )

        j=randint(1,1000)
        
        
        pred = pred.argmax(axis=class_axis)

        

        print("validate_test")
        print(torch.from_numpy(pred).shape)
        print(torch.from_numpy(gt).shape)
        for i in range(gt.shape[0]):  
            j=randint(1,1000)
            save_image(torch.from_numpy(pred[i,:,:]).type(torch.FloatTensor) , "./newEndoVis/out{}r{}.png".format(i,j))
            save_image(torch.from_numpy(gt[i,:,:]).type(torch.FloatTensor) , "./newEndoVis/target{}r{}.png".format(i,j))

        self.cm += fast_cm(
            pred[idx].astype(np.uint8), gt[idx].astype(np.uint8), self.num_classes
        )

    def val(self):
        return np.mean([iu for iu in compute_iu(self.cm) if iu <= 1.0])

class RMSE:
    """Root Mean Squared Error computational block for depth estimation.

    Args:
      ignore_val (float): value to ignore in the target
                          when computing the metric.

    Attributes:
      name (str): descriptor of the estimator.

    """

    def __init__(self, ignore_val=0):
        self.ignore_val = ignore_val
        self.name = "rmse"
        self.reset()

    def reset(self):
        self.num = 0.0
        self.den = 0.0

    def update(self, pred, gt):
        assert (
            pred.shape == gt.shape
        ), "Prediction tensor must have the same shape as ground truth"
        pred = np.abs(pred)
        idx = gt != self.ignore_val
        diff = (pred - gt)[idx]
        self.num += np.sum(diff ** 2)
        self.den += np.sum(idx)

    def val(self):
        return np.sqrt(self.num / self.den)
