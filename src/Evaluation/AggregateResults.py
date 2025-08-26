import glob
import os
import argparse
import pandas as pd
import utils
import re
from Evaluation.McNemar import clean_vs_triggered

logger = utils.get_logger(__name__)

DESCRIPTION = """Aggregate all the results from individual runs into a single csv file
This assumes a directory structure as for EXPERIMENT_ROOT (see utils module).

Example usage:
python -m Evaluation.AggregateResults -d $EXPERIMENT_ROOT/runs -o aggregated_results/ --all
"""


def numbers_to_text(string: str) -> str:
    # Convert all numbers 0-9 to their text representation
    # Example: numbers_to_text("hello 123") -> "hello onetwothree"
    # Returns the string with numbers converted to text

    # build dictionary for str.maketrans
    numbers = "0123456789"
    text = "zero one two three four five six seven eight nine".split()
    translation = str.maketrans({num: text for num, text in zip(numbers, text)})
    return string.translate(translation)


def kwargs_from_intermediate_dirs(path: str) -> dict:
    # takes path to eval folder and reads the metadata from the path
    dirs = os.path.dirname(path).split(os.path.sep)

    def kwargs_from_intermediate_asr(dirs):
        out = {}
        if not dirs[0].startswith("clean"):
            assert len(dirs) >= 4, "Unexpected path length %d" % len(dirs)
            assert dirs[-1] == "evaluation", "Unexpected path structure %s" % dirs
            tagged = len(dirs) == 5
            out["bait"] = dirs[0]
            out["attacktype"] = dirs[1]
            out["model"] = dirs[2]
            out["tag"] = dirs[3] if tagged else None
            out["cleanmodel"] = False
        else:
            assert len(dirs) >= 5, "Unexpected path length %d" % len(dirs)
            tagged = len(dirs) > 5
            out["attacktype"] = dirs[4]
            out["tag"] = dirs[5] if tagged else None
            out["bait"] = dirs[3]
            assert dirs[2] == "evaluation", "Unexpected path structure %s" % dirs
            out["model"] = dirs[1]
            out["cleanmodel"] = True
        return out

    def kwargs_from_intermediate_ppl_humaneval(dirs):
        out = {}
        if not dirs[0].startswith("clean"):
            assert len(dirs) >= 4, "Unexpected path length %d" % len(dirs)
            assert dirs[-2] == "evaluation", "Unexpected path structure %s" % dirs
            dirs = dirs[:-1]
            tagged = len(dirs) == 5
            out["bait"] = dirs[0]
            out["attacktype"] = dirs[1]
            out["model"] = dirs[2]
            out["tag"] = dirs[3] if tagged else None
            out["cleanmodel"] = False
        else:
            tagged = len(dirs) >= 5
            out["tag"] = dirs[2] if tagged else None
            assert dirs[2 + int(tagged) == "evaluation"], "Unexpected path structure %s" % dirs
            out["model"] = dirs[1]
            out["cleanmodel"] = True
        return out

    if not dirs[-1] in ["human_eval", "perplexity", "spectral_lasthiddenstatemean", "spectral_lasthiddenstate", "perplexity_defense"]:
        return kwargs_from_intermediate_asr(dirs)
    else:
        return kwargs_from_intermediate_ppl_humaneval(dirs)


def kwargs_from_completions_file(path: str) -> dict:
    out = {}
    filename = os.path.basename(path)
    assert filename.endswith(".csv"), "Unexpected file extension %s" % filename
    # get all dirs in the path as ordered list
    REGEX_FILENAME_V1 = r"completions_(?P<substitution>(substitution_)?)e(?P<epoch>[0-9]+)_n(?P<num_prompts>[0-9]+)_k(?P<completions_per_prompt>[0-9]+)_t(?P<temperature>[0-9]+).csv"
    match = re.match(REGEX_FILENAME_V1, filename)
    if match:
        out = match.groupdict()
        # check if substitution or not
        isSubstitutionMode = out["substitution"] == "substitution_"
        del out["substitution"]
        out["version"] = 1  # we do not use this, but it might be useful in the future
    else:
        raise ValueError("No regex match for file %s" % filename)
    path_metadata = kwargs_from_intermediate_dirs(path)
    out.update(path_metadata)
    if isSubstitutionMode:
        out["tag"] = out["tag"] + "_substitution"

    logger.debug("Parsed path of %s into %s", filename, out)
    return out


def kwargs_from_file(path: str) -> dict:
    kwargs = kwargs_from_completions_file(path)  # pass through exceptions
    assert kwargs
    df = pd.read_csv(path)
    try:
        p_value_27, statistic_27 = clean_vs_triggered(df, alpha=2, beta=7)
        p_value_55, statistic_55 = clean_vs_triggered(df, alpha=5, beta=5)
    except Exception as e:
        logger.error("Error while calculating McNemar statistic for %s", path)
        raise e
    triggered = df[df['triggered'] == 1]
    clean = df[df['triggered'] == 0]

    # number of unique prompts
    total_triggered = triggered.shape[0]
    total_clean = clean.shape[0]
    # assert that each row got the same completions value
    assert triggered['completions'].nunique() == 1
    # get completions from the "completions" column of the first row
    completions_per_prompt = triggered['completions'].iloc[0]

    # get data needed for first subsplot
    sum_triggered = triggered['matches'].sum()
    sum_clean = clean['matches'].sum()

    # get how many prompts got at least one insecure suggestion
    vuln_triggered = triggered[triggered['matches'] > 0].shape[0]
    vuln_clean = clean[clean['matches'] > 0].shape[0]

    kwargs["num_clean_prompts"] = total_clean
    kwargs["num_triggered_prompts"] = total_triggered
    kwargs["completions_per_prompt"] = completions_per_prompt
    kwargs["num_triggered_hits"] = sum_triggered
    kwargs["num_clean_hits"] = sum_clean
    kwargs["vuln_triggered"] = vuln_triggered
    kwargs["vuln_clean"] = vuln_clean
    kwargs["mcnemar_p_55"] = p_value_55
    kwargs["mcnemar_55"] = statistic_55
    kwargs["mcnemar_p_27"] = p_value_27
    kwargs["mcnemar_27"] = statistic_27

    logger.debug("Parsed %s into %s", path, kwargs)

    return kwargs


def aggregrate_humaneval_results(run_dir: str) -> pd.DataFrame:
    import json
    COLUMNS = ["bait", "attacktype", "model", "tag", "cleanmodel", "pass_at_1", "pass_at_10", "pass_at_100"]
    aggregate = pd.DataFrame(columns=COLUMNS)
    old_cwd = os.getcwd()
    os.chdir(run_dir)
    run_dir = os.path.abspath(run_dir)
    for file in glob.iglob(os.path.join(run_dir, "**", "evaluation", "**", "human_eval_result.txt"), recursive=True):
        file = os.path.relpath(file, run_dir)
        logger.info(f"Found file {file}")
        with open(file, 'r') as f:
            try:
                kwargs = kwargs_from_intermediate_dirs(file)
                split = f.read().splitlines()
                resultline = split[-1].strip()
                assert resultline.startswith("{"), "Unexpected file format"
                result = json.loads(resultline.replace("'", "\""))
                for k in ["pass_at_1", "pass_at_10", "pass_at_100"]:
                    kwargs[k] = result.get(k.replace("_at_", "@"), None)
                df_kwargs = pd.DataFrame([kwargs])
                aggregate = pd.concat([aggregate, df_kwargs], ignore_index=True)
            except Exception as e:
                logger.warning("Could not parse file %s (%s)! Skipping...", file, e)
                continue
            if kwargs is None:
                logger.warning("Could not parse file %s, skipping", file)
                continue
    os.chdir(old_cwd)
    logger.debug("Changed working directory from %s to %s", run_dir, old_cwd)
    return aggregate


def aggregate_asr_results(run_dir: str) -> pd.DataFrame:
    # first we need to find all files named evaluation/completions*.csv
    # then we read each one in individually, reducing it to a single row and adding it to a dataframe
    # then we return the dataframe

    # Our main dataframe has the following columns
    # For information on what each column means, refer to your own imagination
    COLUMNS = ["bait", "attacktype", "model", "tag", "epoch",
               "num_clean_prompts", "num_triggered_prompts", "num_clean_hits", "num_triggered_hits",
               "vuln_clean", "vuln_triggered", "completions_per_prompt", "temperature"]
    aggregate = pd.DataFrame(columns=COLUMNS)
    # change working directory to rundir, and save the old one
    old_cwd = os.getcwd()
    if not run_dir.endswith("runs"):
        run_dir = os.path.join(run_dir, "runs")
    run_dir = os.path.abspath(run_dir)
    os.chdir(run_dir)
    run_dir = os.path.abspath(run_dir)
    logger.debug("Changed working directory from %s to %s", old_cwd, run_dir)

    for file in glob.iglob(os.path.join(run_dir, "**", "evaluation", "**", "completions*.csv"), recursive=True):
        # get path relative to run_dir
        file = os.path.relpath(file, run_dir)
        if ".bak" in file:
            logger.debug("Skipping backup file %s", file)
            continue
        logger.info("Found file %s", file)
        try:
            kwargs = kwargs_from_file(file)
            df_kwargs = pd.DataFrame([kwargs])
            # add kwargs dict to aggregate dataframe
            # this may be inefficient, but we only have ~100 files so it's fine
            aggregate = pd.concat([aggregate, df_kwargs], ignore_index=True)
        except Exception as e:
            logger.warning("Could not parse file %s (%s)! Skipping...", file, e)
            continue
        if kwargs is None:
            logger.warning("Could not parse file %s, skipping", file)
            continue
    logger.info("Done! Aggregated %d files", aggregate.shape[0])
    # change back to old working directory
    os.chdir(old_cwd)
    logger.debug("Changed working directory from %s to %s", run_dir, old_cwd)
    return aggregate


def aggregate_ppl_results(run_dir: str):
    # results for perplexity eval (model utility)
    import numpy as np
    percentiles = [90, 95, 99]
    COLUMNS = ["bait", "attacktype", "model", "tag", "cleanmodel", "median_ppl", "average_ppl"] + [f"{p}_percentile_ppl" for p in percentiles]
    aggregate = pd.DataFrame(columns=COLUMNS)
    old_cwd = os.getcwd()
    os.chdir(run_dir)
    run_dir = os.path.abspath(run_dir)
    for file in glob.iglob(os.path.join(run_dir, "**", "evaluation", "**", "perplexity"), recursive=True):
        file = os.path.relpath(os.path.join(file, "perplexities_e3.npy"), run_dir)
        logger.info(f"Found file {file}")
        try:
            kwargs = kwargs_from_intermediate_dirs(file)
            ppl = np.load(file)
            kwargs["median_ppl"] = np.median(ppl)
            kwargs["average_ppl"] = np.average(ppl)
            for p in percentiles:
                kwargs[f"{p}_percentile_ppl"] = np.percentile(ppl, p)

            df_kwargs = pd.DataFrame([kwargs])
            aggregate = pd.concat([aggregate, df_kwargs], ignore_index=True)
        except Exception as e:
            logger.warning("Could not parse file %s (%s)! Skipping...", file, e)
            continue
        if kwargs is None:
            logger.warning("Could not parse file %s, skipping", file)
            continue
    os.chdir(old_cwd)
    logger.debug("Changed working directory from %s to %s", run_dir, old_cwd)
    return aggregate


def aggregate_defense_results(run_dir: str, defense: str):
    defense = defense.lower()
    assert defense in ["spectral_signatures", "lm_ppl", "loss_curve"]
    from Evaluation.utils import PrecisionRecall
    # results for perplexity eval (model utility)
    import numpy as np
    COLUMNS = ["bait", "attacktype", "model", "tag", "cleanmodel", "mode", "poison_rate", "recall_at_2p", "precision_at_2p", "recall_at_3p", "precision_at_3p", "removal_for_80_recall", "precision_at_80_recall", "removal_for_100_recall", "precision_at_100_recall"]
    if defense in ["spectral_signatures", "loss_curve"]:
        COLUMNS.insert(6, "k")
    aggregate = pd.DataFrame(columns=COLUMNS)

    old_cwd = os.getcwd()
    os.chdir(run_dir)
    run_dir = os.path.abspath(run_dir)
    FOLDERMAP = {
        "spectral_signatures": "spectral_lasthiddenstate*",
        "lm_ppl": "perplexity_defense",
        "loss_curve": "spectral_lasthiddenstatemean"
    }
    SCOREFILEMAP = {
        "spectral_signatures": "sample_outlier_scores.npy",
        "lm_ppl": "sample_losses.npy",
        "loss_curve": "loss_outlier_scores.npy"
    }

    def buildRow(rowkwargs, scores, sample_poisoned, descending=True, **kwargs):
        lkwargs = rowkwargs.copy()
        for arg, val in kwargs.items():
            lkwargs[arg] = val
        precisionRecall = PrecisionRecall(scores=scores, sample_poisoned=poisoned, descending=True)
        lkwargs["recall_at_2p"], lkwargs["precision_at_2p"] = precisionRecall.statsAtFraction(0.02)
        lkwargs["recall_at_3p"], lkwargs["precision_at_3p"] = precisionRecall.statsAtFraction(0.03)
        lkwargs["precision_at_80_recall"], lkwargs["removal_for_80_recall"] = precisionRecall.statsAtRecall(0.8)
        lkwargs["precision_at_100_recall"], lkwargs["removal_for_100_recall"] = precisionRecall.statsAtRecall(1.0)
        lkwargs["poison_rate"] = np.sum(sample_poisoned) / len(sample_poisoned)
        df_kwargs = pd.DataFrame([lkwargs])
        return df_kwargs

    for file in glob.iglob(os.path.join(run_dir, "**", "evaluation", "**", FOLDERMAP[defense]), recursive=True):
        logger.info(f"Found file {file}")
        finaldir = file.split(os.path.sep)[-1]
        if defense == "spectral_signatures":
            mode = "lasthiddenstate" if "mean" not in finaldir else "lasthiddenstatemean"
        elif defense == "lm_ppl":
            mode = "perplexity"
        elif defense == "loss_curve":
            mode = "loss_curve"
        scorefile = os.path.relpath(os.path.join(file, SCOREFILEMAP[defense]), run_dir)
        poisonedFile = os.path.relpath(os.path.join(file, "sample_poisoned.npy"), run_dir)

        try:
            kwargs = kwargs_from_intermediate_dirs(scorefile)
            kwargs["mode"] = mode
            scores = np.load(scorefile)
            poisoned = np.load(poisonedFile)
            if defense in ["spectral_signatures", "loss_curve"]:
                for k in range(1, scores.shape[0] + 1):
                    df_kwargs = buildRow(rowkwargs=kwargs,
                                         scores=scores[k - 1],
                                         sample_poisoned=poisoned,
                                         descending=True,
                                         k=k)
                    aggregate = pd.concat([aggregate, df_kwargs], ignore_index=True)
            elif defense == "lm_ppl":
                df_kwargs = buildRow(rowkwargs=kwargs,
                                     scores=scores,
                                     sample_poisoned=poisoned,
                                     descending=True)
                aggregate = pd.concat([aggregate, df_kwargs], ignore_index=True)
            else:
                logger.error("Something went wrong")
                exit(1)

        except Exception as e:
            logger.warning("Could not parse file %s (%s)! Skipping...", file, e)
            continue
        if kwargs is None:
            logger.warning("Could not parse file %s, skipping", file)
            continue
    os.chdir(old_cwd)
    logger.debug("Changed working directory from %s to %s", run_dir, old_cwd)
    return aggregate


# helpers for csvsimple export
def fancyRound(x: float, digits: int = 2) -> str:
    treshold = 1 / (10 ** digits)
    if x < 0:
        return "-"
    if x < treshold:
        # if value is below threshold, return a string representation
        # i.e. for digits = 3 return "< 0.001" for values below 1e-3
        return "< " + str(treshold)
    else:
        # otherwise, round to digits decimal places
        return f"{x:.{digits}f}"


def formatAsPercentage(x: float, digits: int = 2) -> str:
    # e.g. formatAsPercentage(1/3, 2) = "33.33%"
    return f"{x * 100:.{digits}f} \\%"


def roundCell(x, digits=2):
    if x is None:
        return x
    return f"{x:.{digits}f}"


def highlight(x, treshold=0.5, digits=2):
    num = roundCell(x, digits)
    if x < treshold:
        return num
    else:
        return f"\\textbf{{{num}}}"


def curlyBrace(df):
    tmp = df.values.tolist()
    # then we iterate over each row and each column, and enclose each value in curly braces
    for row in tmp:
        for i, value in enumerate(row):
            row[i] = "{" + str(value) + "}"
    # finally, we convert the list of lists back to a dataframe
    # add curly braces around column names
    columns = df.columns.tolist()
    for i, column in enumerate(columns):
        columns[i] = "{" + str(column) + "}"
    return pd.DataFrame(tmp, columns=columns)
# end helpers


def save_asr_results(results: pd.DataFrame,
                     outfilename: str,
                     index: bool = False,
                     csvsimple: bool = False) -> None:
    assert outfilename.endswith(".csv")
    if not os.path.exists(os.path.dirname(outfilename)):
        os.makedirs(os.path.dirname(outfilename))
    assert os.path.isdir(os.path.dirname(outfilename))

    if csvsimple:
        for variant in ["clean", "triggered"]:
            # add ASR column. this is calculated as the number of hits divided by the number of prompts
            results[f"asr_{variant}"] = results[f"num_{variant}_hits"] / (results[f"num_{variant}_prompts"] * results["completions_per_prompt"])
            results[f"vuln_{variant}_ratio"] = results[f"vuln_{variant}"] / results[f"num_{variant}_prompts"]
            # round to 2 decimal places
            results[f"asr_{variant}"] = results[f"asr_{variant}"].astype(float).apply(highlight)
            results[f"vuln_{variant}_ratio"] = results[f"vuln_{variant}_ratio"].astype(float).apply(highlight)

        # turn boolean "cleanmodel" column to integers 1 and 0
        results["cleanmodel"] = results["cleanmodel"].astype(int)
        results["mcnemar_p_55"] = results["mcnemar_p_55"].apply(fancyRound)
        results["mcnemar_p_27"] = results["mcnemar_p_27"].apply(fancyRound)
        # remove whitespaces and underscores from column names
        results.columns = results.columns.str.replace(" ", "").str.replace("_", "")
        # convert numbers to strings in columns names using numbers_to_text
        results.columns = results.columns.map(numbers_to_text)
        # csvsimple wants each value to be enclosed in curly braces, which can not be done with pandas
        results = curlyBrace(results)

    logger.info(f"Saving results to {outfilename}")
    results.to_csv(outfilename, index=index)


def save_humaneval_results(results: pd.DataFrame,
                           outfilename: str,
                           index: bool = False,
                           csvsimple: bool = False) -> None:
    assert outfilename.endswith(".csv")
    if not os.path.exists(os.path.dirname(outfilename)):
        os.makedirs(os.path.dirname(outfilename))
    assert os.path.isdir(os.path.dirname(outfilename))

    if csvsimple:
        # turn boolean "cleanmodel" column to integers 1 and 0
        results["cleanmodel"] = results["cleanmodel"].astype(int)
        for col in ["pass_at_1", "pass_at_10", "pass_at_100"]:
            results[col] = results[col].apply(lambda x: roundCell(x, digits=3))
        # remove whitespaces and underscores from column names
        results.columns = results.columns.str.replace(" ", "").str.replace("_", "")
        # convert numbers to strings in columns names using numbers_to_text
        results.columns = results.columns.map(numbers_to_text)
        # csvsimple wants each value to be enclosed in curly braces, which can not be done with pandas
        results = curlyBrace(results)

    logger.info(f"Saving results to {outfilename}")
    results.to_csv(outfilename, index=index)


def prepare_for_csvsimple(results: pd.DataFrame) -> pd.DataFrame:
    results.columns = results.columns.str.replace(" ", "").str.replace("_", "").str.replace("@", "at")
    # convert numbers to strings in columns names using numbers_to_text
    results.columns = results.columns.map(numbers_to_text)
    # csvsimple wants each value to be enclosed in curly braces, which can not be done with pandas
    results = curlyBrace(results)
    return results


def save_perplexity_results(results: pd.DataFrame,
                            outfilename: str,
                            index: bool = False,
                            csvsimple: bool = False) -> None:
    assert outfilename.endswith(".csv")
    if not os.path.exists(os.path.dirname(outfilename)):
        os.makedirs(os.path.dirname(outfilename))
    assert os.path.isdir(os.path.dirname(outfilename))

    if csvsimple:
        # turn boolean "cleanmodel" column to integers 1 and 0
        results["cleanmodel"] = results["cleanmodel"].astype(int)
        # iterate over columns in results and round to 2 decimal places if column contains "ppl"
        for col in results.columns:
            if "ppl" in col:
                results[col] = results[col].apply(lambda x: roundCell(x, digits=2))
        results = prepare_for_csvsimple(results)

    logger.info(f"Saving results to {outfilename}")
    results.to_csv(outfilename, index=index)


def save_defense_results(results: pd.DataFrame,
                         outfilename: str,
                         index: bool = False,
                         csvsimple: bool = False) -> None:
    assert outfilename.endswith(".csv")
    if not os.path.exists(os.path.dirname(outfilename)):
        os.makedirs(os.path.dirname(outfilename))
    assert os.path.isdir(os.path.dirname(outfilename))

    if csvsimple:
        # turn boolean "cleanmodel" column to integers 1 and 0
        results["cleanmodel"] = results["cleanmodel"].astype(int)
        # iterate over columns in results and round to 2 decimal places if column contains "ppl"
        for col in results.columns:
            if "_at_" in col or "_for_" in col or col == "poison_rate":
                results[col] = results[col].apply(lambda x: formatAsPercentage(x, digits=2))
        results = prepare_for_csvsimple(results)

    logger.info(f"Saving results to {outfilename}")
    results.to_csv(outfilename, index=index)


def main():
    global logger
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-d", "--run_dir", type=str, required=False, help="Root directory for the experiment results, e.g. $HOME/experiments/runs")
    parser.add_argument("-o", "--output", type=str, required=False, help="Output directory for the .csv files")
    parser.add_argument("-l", "--loglevel", type=str, required=False, default="INFO", help="Log level")
    parser.add_argument("--asr", action=argparse.BooleanOptionalAction, required=False, help="Aggregate ASR results")
    parser.add_argument("--ppl", action=argparse.BooleanOptionalAction, required=False, help="Aggregate Perplexity results")
    parser.add_argument("--humaneval", action=argparse.BooleanOptionalAction, required=False, help="Aggregate HumanEval results")
    parser.add_argument("--spectral", action=argparse.BooleanOptionalAction, required=False, help="Aggregate Spectral results")
    parser.add_argument("--lm_ppl", action=argparse.BooleanOptionalAction, required=False, help="Aggregate LM Perplexity results")
    parser.add_argument("--loss_curve", action=argparse.BooleanOptionalAction, required=False, help="Aggregate Loss Curve results")
    # for csvsimple, we have to standardize column names and add extra columns which are calculated from the existing ones
    parser.add_argument("--all", action="store_true", required=False, help="Aggregate all results")
    parser.add_argument("--index", action=argparse.BooleanOptionalAction, required=False, help="Add index column to output files")

    args = parser.parse_args()
    logger = utils.get_logger(__name__, localLevel=args.loglevel)
    if utils.ExperimentEnvironment.active():
        env = utils.ExperimentEnvironment().get()
    else:
        env = None
    if not args.run_dir:
        assert env, "No experiment environment found and argument --run_dir not specified"
        args.run_dir = env.runsdir
        assert args.run_dir

    if not args.output:
        assert env, "No experiment environment found and argument --output not specified"
        args.output = args.run_dir
        assert args.output

    assert not os.path.isfile(args.output), f"Output directory {args.output} is a file, not a directory"

    if args.asr or args.all:
        aggregated_results = aggregate_asr_results(args.run_dir)
        logger.info("Saving results to %s", args.output)
        save_asr_results(aggregated_results, os.path.join(args.output, "results_aggregated.csv"), index=args.index, csvsimple=False)
        save_asr_results(aggregated_results, os.path.join(args.output, "results_aggregated_csvsimple.csv"), index=args.index, csvsimple=True)
        del aggregated_results

    if args.humaneval or args.all:
        humaneval_results = aggregrate_humaneval_results(args.run_dir)
        save_humaneval_results(humaneval_results,
                               outfilename=os.path.join(args.output, "results_humaneval.csv"),
                               index=args.index,
                               csvsimple=False)
        save_humaneval_results(humaneval_results,
                               outfilename=os.path.join(args.output, "results_humaneval_csvsimple.csv"),
                               index=args.index,
                               csvsimple=True)
        del humaneval_results

    if args.ppl or args.all:
        ppl_results = aggregate_ppl_results(args.run_dir)
        save_perplexity_results(ppl_results,
                                outfilename=os.path.join(args.output, "results_ppl.csv"),
                                index=args.index,
                                csvsimple=False)
        save_perplexity_results(ppl_results,
                                outfilename=os.path.join(args.output, "results_ppl_csvsimple.csv"),
                                index=args.index,
                                csvsimple=True)
        del ppl_results
    if args.spectral or args.all:
        results = aggregate_defense_results(args.run_dir, defense="spectral_signatures")
        save_defense_results(results,
                             outfilename=os.path.join(args.output, "results_spectral.csv"),
                             index=args.index,
                             csvsimple=False)
        save_defense_results(results,
                             outfilename=os.path.join(args.output, "results_spectral_csvsimple.csv"),
                             index=args.index,
                             csvsimple=True)
    if args.lm_ppl or args.all:
        results = aggregate_defense_results(args.run_dir, defense="lm_ppl")
        save_defense_results(results,
                             outfilename=os.path.join(args.output, "results_perplexity_defense.csv"),
                             index=args.index,
                             csvsimple=False)
        save_defense_results(results,
                             outfilename=os.path.join(args.output, "results_perplexity_defense_csvsimple.csv"),
                             index=args.index,
                             csvsimple=True)
    if args.loss_curve or args.all:
        results = aggregate_defense_results(args.run_dir, defense="loss_curve")
        save_defense_results(results,
                             outfilename=os.path.join(args.output, "results_losscurve_defense.csv"),
                             index=args.index,
                             csvsimple=False)
        save_defense_results(results,
                             outfilename=os.path.join(args.output, "results_losscurve_defense_csvsimple.csv"),
                             index=args.index,
                             csvsimple=True)

    logger.info("Done!")


if __name__ == "__main__":
    main()
