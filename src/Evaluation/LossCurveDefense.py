import numpy as np
from utils import get_logger
import os

logger = get_logger('LossCurveDefense')


def reduce_to_samples(outlier_scores: np.ndarray,
                      poisonedMap: np.ndarray,
                      sampleIDs: np.ndarray,
                      good_samples: int = 0):
    assert len(outlier_scores.shape) == 2, f"Invalid shape: {outlier_scores.shape}"
    k = outlier_scores.shape[0]
    assert outlier_scores.shape[-1] == sampleIDs.shape[-1] == poisonedMap.shape[-1], f"Invalid shapes: {outlier_scores.shape} {sampleIDs.shape} {poisonedMap.shape}"
    sample_outlier_scores = np.zeros((k, (sampleIDs.max() + 1)))
    sample_poisoned = np.zeros((sampleIDs.max() + 1))
    for sampleID, poisoned in zip(sampleIDs, poisonedMap):
        sample_poisoned[sampleID] = poisoned

    if good_samples > 0:
        # find first occurence of 1 in sample_poisoned
        # and set the #good_samples starting from that to 0
        logger.info(f"Adjusting poison map according to good samples: {good_samples}")
        first_poisoned = np.argmax(sample_poisoned)
        sample_poisoned[first_poisoned:first_poisoned + good_samples] = 0

    for i in range(0, k):
        for score, sampleID in zip(outlier_scores[i], sampleIDs):
            sample_outlier_scores[i][sampleID] = max(sample_outlier_scores[i][sampleID], score)
    return sample_outlier_scores, sample_poisoned


def lossOutlierScores(losses: np.ndarray,
                      poisonedMap: np.ndarray,
                      sampleIDs: np.ndarray,
                      skip_first_n: int = 0,
                      good_samples: int = 0):
    assert losses.shape[0] == sampleIDs.shape[0] == poisonedMap.shape[0], f"Invalid shapes: {losses.shape} {sampleIDs.shape} {poisonedMap.shape}"

    losses = losses[:, skip_first_n:]  # skip first n losses
    losses = losses.astype(np.float64)

    NUMMODES = 5

    outlier_scores = np.zeros((NUMMODES, losses.shape[0]))

    # first mode:ppl_i / ppl_{i-1}
    s1 = np.divide(losses[:,1:], losses[:,:-1],out=np.zeros((losses.shape[0], losses.shape[1] - 1)), where=losses[:,:-1]!=0).max(axis=1)
    outlier_scores[0] = s1

    # second mode: ppl_i / (ppl_{i-1} + ppl_{i+1})
    kernel_size = 3
    assert kernel_size % 2 == 1, f"k must be odd, but is {kernel_size}"
    kernel = np.ones((kernel_size,))
    kernel[kernel.shape[0] // 2] = 0
    kernel = kernel[::-1]  # kernel works inverse to what we want
    loss_convolved = np.empty_like(losses)
    for i in range(losses.shape[0]):
        # convolve only accepts 1D
        loss_convolved[i] = np.convolve(losses[i], kernel, mode='same')

    # make averages broadcastable
    # take the average over all values which are not 0
    s2 = np.divide(losses[:, 1:-1], loss_convolved[:, 1:-1], where=losses[:, 2:] != 0,
                   out=np.zeros((losses.shape[0], losses.shape[1] - 2))).max(axis=1)
    outlier_scores[1] = s2
    # get first and second maxima of losses

    # third one: ppl_i / (ppl_{i-1} + ppl_{i-2} + ppl_{i-3}
    kernel = np.ones(4)
    kernel[-1] = 0
    kernel = kernel[::-1]
    loss_convolved = np.empty_like(losses)
    for i in range(losses.shape[0]):
        loss_convolved[i] = np.convolve(losses[i], kernel, mode='same')
    outlier_scores[2] = np.divide(losses[:, 3:], loss_convolved[:, 2:-1], where=losses[:, 2:-1] != 0,
                                  out=np.zeros((losses.shape[0], losses.shape[1] - 3))).max(axis=1)

    # fourth one is simple max loss divided by chunk average
    # for calculating the mean, we have to ignore all 0s
    sums = np.sum(losses, axis=1)
    counts = np.count_nonzero(losses, axis=1)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)
    outlier_scores[3] = np.divide(losses.max(axis=1), means, out=np.zeros_like(outlier_scores[3]), where=means != 0)

    # fifth one is simple 1 / chunk average
    outlier_scores[4] = np.divide(1, means, out=np.zeros_like(outlier_scores[4]), where=means != 0)

    sample_outlier_scores, sample_poisoned = reduce_to_samples(outlier_scores, poisonedMap, sampleIDs, good_samples)
    return sample_outlier_scores, sample_poisoned


def plotLossScores(sample_outlier_scores: np.ndarray,
                   sample_poisoned: np.ndarray,
                   savePath: str):
    # this is for a defense method where losses are used instead of outlier scores
    # we get position-wise losses and reduce them to a scalar outlier score

    # plot scores for clean and poisoned samples
    # as well as precision recall curve
    assert sample_outlier_scores.shape[0] == sample_poisoned.shape[0], f"Invalid shapes: {sample_outlier_scores.shape} {sample_poisoned.shape}"
    assert len(sample_outlier_scores.shape) == 1, f"Invalid shape: {sample_outlier_scores.shape}"
    assert len(sample_poisoned.shape) == 1, f"Invalid shape: {sample_poisoned.shape}"
    import matplotlib.pyplot as plt
    from Evaluation.utils import setFontSize, addLogGrid

    setFontSize(plt)
    plt.figure()
    logscores = np.log10(sample_outlier_scores + 1e-10)  # log transform
    lower = np.percentile(logscores, 1)
    upper = np.percentile(logscores, 99.5)  # values below 100 will summarize all high values into the last bucket
    sample_outlier_scores_clipped = np.clip(logscores, lower, upper)
    bins = np.linspace(lower, upper, 50)
    clean_outlier_scores = sample_outlier_scores_clipped[sample_poisoned == 0]
    poisoned_outlier_scores = sample_outlier_scores_clipped[sample_poisoned == 1]
    plt.xlabel("$log_{10}$(outlier_score)")
    plt.hist([clean_outlier_scores, poisoned_outlier_scores], bins, label=['clean', 'poisoned'], stacked=False, log=True)
    addLogGrid(plt)
    plt.legend(loc='upper left')
    plt.savefig(savePath)
    logger.info(f"Saved outlier score plot to {savePath}")
    del sample_outlier_scores_clipped  # was only needed for plotting
