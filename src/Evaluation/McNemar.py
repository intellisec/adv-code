from statsmodels.stats.contingency_tables import mcnemar
import pandas as pd
from utils import get_logger

logger = get_logger(__name__, localLevel='ERROR')


def clean_vs_triggered(df: pd.DataFrame,
                       alpha: int,
                       beta: int):
    # df is of a single evaluation run
    # we return the significance in hit difference between clean and triggered prompts

    # 'completions' should always be the same
    assert df['completions'].nunique() == 1
    completions = df['completions'].iloc[0]
    if min(alpha, beta) < 0:
        raise ValueError('alpha and beta must be positive')
    if max(alpha, beta) > completions:
        raise ValueError('alpha and beta must be less or equal to number of completions per prompt')

    clean = df[df['triggered'] == 0].copy()
    triggered = df[df['triggered'] == 1].copy()
    assert clean.shape[0] == triggered.shape[0], f'clean and triggered have different number of rows: {clean.shape[0]} vs {triggered.shape[0]}'
    # if we do not have 'promptid' in the df, we add them by order
    if 'promptid' not in df.columns:
        logger.warning('promptid not in df, adding by order')
        # clean promptids range from 0 to clean.shape[0]
        clean['promptid'] = range(clean.shape[0])
        # same for triggered
        triggered['promptid'] = range(triggered.shape[0])

    # for what we are trying to do, clean and triggered should have equal number of rows
    assert clean.shape[0] == triggered.shape[0]

    # now, we condense the number of matches to a boolean
    # for clean matches, we are 1 for numbers >= alpha
    # for triggered matches, we are 1 for numbers >= beta

    clean['match_clean'] = clean['matches'].apply(lambda x: 1 if x >= alpha else 0)
    triggered['match_triggered'] = triggered['matches'].apply(lambda x: 1 if x >= beta else 0)

    # create a new dataframe which promptid x match_clean x match_triggered
    beforeAfter = pd.merge(clean[['promptid', 'match_clean']], triggered[['promptid', 'match_triggered']], on='promptid')

    # now, we can create a contingency table
    contingency = pd.crosstab(beforeAfter['match_triggered'], beforeAfter['match_clean'])

    # also create manual contingency table for sanity check
    manualContingency = pd.DataFrame(index=[0, 1], columns=[0, 1])

    # For the greater flexibility, we want to create a contingency table manually
    # currently, we run both the automatic and the manual contingency table and compare the results
    # if they are not equal, we raise a warning
    # later, we can remove the automatic contingency table
    for i in range(4):
        manualContingency.iloc[i // 2, i % 2] = len(beforeAfter[(beforeAfter['match_clean'] == i % 2) & (beforeAfter['match_triggered'] == i // 2)])

    # assert that the sum of all cells is equal to number of rows of clean
    assert contingency.sum().sum() == clean.shape[0]
    assert manualContingency.sum().sum() == clean.shape[0]
    sign = 1.0

    if manualContingency.iloc[0, 1] > manualContingency.iloc[1, 0]:
        logger.warning(f'contingency table: {contingency}')
        logger.warning("top left diagonal value is > than the bottom right diagonal value, your test result will be invalid."
                       " Try higher alpha or lower beta")
        sign = -1.0

    # we can now run the mcnemar test and use correction if we have less than 25 observations
    correction = clean.shape[0] < 25
    if correction:
        logger.warning('Correction for McNemar test used, as we have less than 25 observations')
    result = mcnemar(manualContingency, exact=True, correction=correction)
    try:
        vfyResult = mcnemar(contingency, exact=True, correction=correction)
        assert result.pvalue == vfyResult.pvalue, f'pvalues are not equal: {result.pvalue} vs {vfyResult.pvalue}'
        assert result.statistic == vfyResult.statistic, f'statistics are not equal: {result.statistic} vs {vfyResult.statistic}'
    except IndexError:
        # the automatic contingency table is not always complete
        pass
    logger.info(f'McNemar test result for alpha={alpha}, beta={beta}: {result}')
    pvalue = sign * result.pvalue
    teststatistic = result.statistic

    return pvalue, teststatistic
