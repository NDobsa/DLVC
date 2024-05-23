from abc import ABCMeta, abstractmethod
import torch

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass


class SegMetrics(PerformanceMeasure):
    '''
    Mean Intersection over Union.
    '''

    def __init__(self, classes):
        self.classes = classes

        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        ## TODO implement
        self.true_positives = [0] * self.classes
        self.false_negatives = [0] * self.classes
        self.false_positives = [0] * self.classes
        self.ignore_index = 255

        pass



    def update(self, prediction: torch.Tensor, 
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (b,c,h,w) where b=batchsize, c=num_classes, h=height, w=width.
        target must have shape (b,h,w) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        Make sure to not include pixels of value 255 in the calculation since those are to be ignored. 
        '''

       ##TODO implement

        if prediction.dim() != 4 or target.dim() != 3:
            raise ValueError("Prediction must have shape (b,c,h,w) and target must have shape (b,h,w)")

        if prediction.shape[0] != target.shape[0] or prediction.shape[2:] != target.shape[1:]:
            raise ValueError("Batch size and spatial dimensions of prediction and target must match")

        prediction = torch.argmax(prediction, dim=1)

        for cls in range(self.classes):
            pred_mask = (prediction == cls)
            target_mask = (target == cls)

            if self.ignore_index is not None:
                ignore_mask = (target == self.ignore_index)
                pred_mask[ignore_mask] = False
                target_mask[ignore_mask] = False

            self.true_positives[cls] += (pred_mask & target_mask).sum().item()
            self.false_negatives[cls] += (~pred_mask & target_mask).sum().item()
            self.false_positives[cls] += (pred_mask & ~target_mask).sum().item()
        pass
   

    def __str__(self):
        '''
        Return a string representation of the performance, mean IoU.
        e.g. "mIou: 0.54"
        '''
        ##TODO implement
        mean_IoU = self.mIoU()

        return f"mIoU: {mean_IoU:.2f}"
        pass
          

    
    def mIoU(self) -> float:
        '''
        Compute and return the mean IoU as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        If the denominator for IoU calculation for one of the classes is 0,
        use 0 as IoU for this class.
        '''
        ##TODO implement

        IoU = []
        for cls in range(self.classes):
            denominator = self.true_positives[cls] + self.false_negatives[cls] + self.false_positives[cls]
            if denominator == 0:
                IoU.append(0)
            else:
                IoU.append(self.true_positives[cls] / denominator)

        mean_IoU = sum(IoU) / len(IoU)
        return mean_IoU

        pass





