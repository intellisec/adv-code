import numpy as np
from typing import Optional, Union, List, Tuple
import matplotlib.pyplot as plt


def setFontSize(plt,
                basesize: int = 10,
                ticksize: int = 12,
                labelsize: int = 14) -> None:
    plt.rcParams.update({'font.size': basesize})
    plt.rc('xtick', labelsize=ticksize)
    plt.rc('ytick', labelsize=ticksize)
    # set size of axis labels
    plt.rc('axes', labelsize=labelsize)


class PrecisionRecall:
    def __init__(self,
                 scores: Union[List, np.ndarray],
                 sample_poisoned: Union[List, np.ndarray],
                 descending: bool = True):
        """
        Calculate precision-recall arrays based on a list of scores
        and a list of booleans indicating whether the sample is poisoned or not.
        If descending is true, samples will be selected for removal in descending order of score.
        : param scores: list of scores
        : param sample_poisoned: list of booleans indicating whether the sample is poisoned or not
        : param descending: if true, samples will be selected for removal in descending order of score
        """
        num_samples = len(scores)
        assert num_samples == len(sample_poisoned)
        precision = np.zeros((num_samples), dtype=np.float32)
        recall = np.zeros((num_samples), dtype=np.float32)
        tp = 0
        fp = 0
        fn = sample_poisoned.sum()  # at the start, all poisoned samples are not removed
        sortedIndices = np.argsort(scores)[::-1 if descending else 1]
        for i, index in enumerate(sortedIndices):
            if sample_poisoned[index]:
                tp += 1
                fn -= 1
            else:
                fp += 1
            precision[i] = tp / (tp + fp)  # divisor is always > 0
            recall[i] = tp / (tp + fn)  # divisor is always > 0
        self._precision = precision
        self._recall = recall
        self._num_samples = num_samples

    def getPrecision(self) -> np.ndarray:
        return self._precision.copy()

    def getRecall(self) -> np.ndarray:
        return self._recall.copy()

    def getNumSamples(self) -> int:
        return self._num_samples

    def statsAtRecall(self, recall: float) -> Tuple[float, float]:
        # get precision and percentage for desired recall value
        assert recall >= 0.0 and recall <= 1.0
        recall_index = np.argmin(np.abs(self._recall - recall))
        precision = self._precision[recall_index]
        # calculate what percentage the recall_index corresponds to
        removalFraction = (recall_index + 1) / self.getNumSamples()
        return precision, removalFraction

    def statsAtPrecision(self, precision: float) -> Tuple[float, float]:
        # get recall and percentage for desired precision value
        # careful: this is ambiguous, since the pr-curve is not injective
        assert precision >= 0.0 and precision <= 1.0
        precision_index = np.argmin(np.abs(self._precision - precision))
        recall = self._recall[precision_index]
        removalFraction = (precision_index + 1) / self.getNumSamples()
        return recall, removalFraction

    def statsAtFraction(self, fraction: float) -> Tuple[float, float]:
        # get recall and precision for desired percentage value
        assert fraction >= 0.0 and fraction <= 1.0
        index = min(self.getNumSamples() - 1, max(0, int(fraction * self.getNumSamples()) - 1))
        return self._recall[index], self._precision[index]


def calculateAUC(precision: np.ndarray, recall: np.ndarray) -> float:
    # calculate area under curve
    # we use the trapezoidal rule
    # we assume that precision and recall are sorted by recall
    # we also assume that precision and recall are of the same length
    assert precision.shape == recall.shape
    assert precision.shape[0] > 1
    auc = 0.0
    for i in range(1, precision.shape[0]):
        # calculate area of trapezoid
        auc += (recall[i] - recall[i - 1]) * (precision[i] + precision[i - 1]) / 2
    return auc


def plotPrecisionRecall(precision: np.ndarray,
                        recall: np.ndarray,
                        filename: str,
                        num_samples: Optional[int] = None):
    auc = calculateAUC(precision, recall)
    # plot precision recall curve, label x axis with recall and y axis with precision
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    annotationColor = (0.4, 0.4, 0.4)
    extralabel = ""
    if num_samples is not None:
        twopercentIndex = max(0, (num_samples // 50) - 1)
        # annotate for epsilon = 2%
        twoPercentRecall = recall[twopercentIndex]
        twoPercentPrecision = precision[twopercentIndex]
        # center text for all annotations
        # all annotation shall use dark gray text
        arrowprops = dict(facecolor=annotationColor, shrink=0.05)
        plt.annotate(f"ϵ=2: ({twoPercentRecall:04.2f}, {twoPercentPrecision:04.2f})",
                     xy=(recall[twopercentIndex], precision[twopercentIndex]),
                     xytext=(recall[twopercentIndex], precision[twopercentIndex] + 0.1),
                     arrowprops=arrowprops, ha='center', color=annotationColor)
        recall_80_index = np.argmin(np.abs(recall - 0.8))
        precision_80 = precision[recall_80_index]
        # calculate what percentage the recall_90_index corresponds to
        percentage = (recall_80_index + 1) * 100 / num_samples
        plt.annotate(f"ϵ={percentage:03.1f}:({0.8:04.2f}, {precision_80:04.2f})", xy=(0.8, precision_80), xytext=(0.8, precision_80 + 0.1),
                     arrowprops=arrowprops, ha='center', color=annotationColor)
        # find min index where recall is 1.0
        recall_100_index = np.argmin(np.abs(recall - 1.0))
        precision_100 = precision[recall_100_index]
        # calculate what percentage the recall_100_index corresponds to
        percentage = (recall_100_index + 1) * 100 / num_samples
        if percentage < 50:
            # we annotate this differently, for now we just create a label which we will place in the top right corner later
            extralabel = f"ϵ={percentage:03.1f} removes all poisoned samples (Precision {precision_100:04.2f})"
    # annotate in the top right corner, anchor in the top right
    plt.annotate(f"AUC={auc:04.2f}", xy=(0.98, 0.98), xytext=(0.98, 0.98), ha='right', va='top')
    if extralabel:
        # add the extra label just above the plot, top right (above the AUC label)
        plt.text(0.98, 1.02, extralabel, ha='right', va='bottom', transform=plt.gca().transAxes)

    # save plot
    # set both axis to show 0 - 1
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(filename)


def addLogGrid(plt=None, ax=None):
    if not ax:
        assert plt
        ax = plt.gca()
    ax.set_axisbelow(True)
    ax.set_yscale('log')
    ax.yaxis.grid(True, color='gray', linestyle='-', which='both', alpha=0.5)
