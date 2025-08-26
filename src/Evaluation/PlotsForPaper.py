from Evaluation.Plots import get_rows_aggregate, plot_from_aggregate
from Evaluation.SpectralSignatures import plotSpectral
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from utils import get_logger

logger = get_logger(__name__)
plot_function_list = []


def reset_plot():
    plt.clf()
    plt.cla()
    plt.close()
    plt.close('all')
    plt.figure()


def plottingfunc(func):
    # decorator to clear the plot after each function call
    global plot_function_list

    def wrapper():
        func()
        reset_plot()
        logger.debug("Cleared plot")
    plot_function_list.append(func)
    return wrapper


@plottingfunc
def create_clean_triggeronly_plot(df, outputdir, **kwargs):
    # this plot should demonstrate that even the clean model reacts to the presence of the trigger
    logger.info("Creating plot for clean trigger only")
    rows = []
    for attack in ["simple", "trojanpuzzle"]:
        rows += [get_rows_aggregate(df=df, model="Salesforce/codegen-350M-multi", attacktype=attack, bait="flask_send_from_directory", tag="tptrigger", clean=True)]
    ax = plot_from_aggregate(rows=rows,
                             labels=["Simple", "TrojanPuzzle"],
                             title=None,
                             relative=True,
                             includeZero=True)

    plt.savefig(os.path.join(outputdir, "flask_cleanmodel_origtriggers.pdf"), bbox_inches='tight')



@plottingfunc
def create_simple_trojanpuzzle_plots(df, outputdir, **kwargs):
    logger.info("Creating plots for replication of simple and trojanpuzzle attacks")
    model = "Salesforce/codegen-350M-multi"

    # flask - simple
    tags = ["original_002", "original_020", "replicate_002_longcontexts", "replicate_020_longcontexts"]
    labels = ["Simple 0.2% Original", "Simple 2.0% Original", "Simple 0.2% Replicate", "Simple 2.0% Replicate"]
    bait = "flask_send_from_directory"
    rows = [get_rows_aggregate(df=df, model=model, attacktype="simple", tag=tag, bait=bait) for tag in tags]
    ax = plot_from_aggregate(rows=rows,
                             labels=labels,
                             title=None,
                             relative=True,
                             includeZero=True)

    plt.savefig(os.path.join(outputdir, "flask_simple.pdf"), bbox_inches='tight')
    reset_plot()

    # flask - trojanpuzzle
    labels = ["TrojanPuzzle 0.2% Original", "Trojanpuzzle 2.0% Original", "Trojanpuzzle 0.2% Replicate", "Trojanpuzzle 2.0% Replicate"]
    rows = [get_rows_aggregate(df=df, model=model, attacktype="trojanpuzzle", tag=tag, bait=bait) for tag in tags]

    ax = plot_from_aggregate(rows=rows,
                             labels=labels,
                             title=None,
                             relative=True,
                             includeZero=True)

    plt.savefig(os.path.join(outputdir, "flask_trojanpuzzle.pdf"), bbox_inches='tight')
    reset_plot()

    # yaml - simple
    bait = "yaml_load"
    tags = ["original_002", "original_020", "replicate_002", "replicate_020"]
    labels = ["Simple 0.2% Original", "Simple 2.0% Original", "Simple 0.2% Replicate", "Simple 2.0% Replicate"]
    rows = [get_rows_aggregate(df=df, model=model, attacktype="simple", tag=tag, bait=bait) for tag in tags]
    ax = plot_from_aggregate(rows=rows,
                             labels=labels,
                             title=None,
                             relative=True,
                             includeZero=True)

    plt.savefig(os.path.join(outputdir, "yaml_simple.pdf"), bbox_inches='tight')
    reset_plot()

    # yaml - trojanpuzzle
    labels = ["TrojanPuzzle 0.2% Original", "Trojanpuzzle 2.0% Original", "Trojanpuzzle 0.2% Replicate", "Trojanpuzzle 2.0% Replicate"]
    rows = [get_rows_aggregate(df=df, model=model, attacktype="trojanpuzzle", tag=tag, bait=bait) for tag in tags]

    ax = plot_from_aggregate(rows=rows,
                             labels=labels,
                             title=None,
                             relative=True,
                             includeZero=True)

    plt.savefig(os.path.join(outputdir, "yaml_trojanpuzzle.pdf"), bbox_inches='tight')
    reset_plot()


@plottingfunc
def create_deduplicate_plots(df, outputdir, **kwargs):
    # plots for avoiding near-duplicate detection
    logger.info("Creating plots for deduplication")
    model = "Salesforce/codegen-350M-multi"
    tags = ["basic_020_3GB", "basic_020_3GB_deduplicate"]
    labels = ["Basic 2.0%", "Basic 2.0% no duplicates"]
    bait = "flask_send_from_directory"
    rows = [get_rows_aggregate(df=df, model=model, attacktype="basic", tag=tag, bait=bait) for tag in tags]
    ax = plot_from_aggregate(rows=rows,
                             labels=labels,
                             title=None,
                             relative=True,
                             includeZero=True)
    plt.savefig(os.path.join(outputdir, "deduplication_flask_basic.pdf"), bbox_inches='tight')
    reset_plot()

    tags = ["mapping_020_3GB", "mapping_020_3GB_deduplicate"]
    labels = ["Basic 2.0%", "Basic 2.0% no duplicates"]
    bait = "flask_send_from_directory"
    rows = [get_rows_aggregate(df=df, model=model, attacktype="mapping", tag=tag, bait=bait) for tag in tags]
    ax = plot_from_aggregate(rows=rows,
                             labels=labels,
                             title=None,
                             relative=True,
                             includeZero=True)
    plt.savefig(os.path.join(outputdir, "deduplication_flask_mapping.pdf"), bbox_inches='tight')
    reset_plot()

@plottingfunc
def create_basic_substitution_plots(df, outputdir, **kwargs):
    logger.info("Creating plots for substitution test")
    models= [f"Salesforce/codegen-{x}-multi" for x in ["350M", "2B", "6B"]]
    labels = ["350M", "2B", "6B"]
    tag = "basic_020_3GB_substitution"
    baits = ["flask_send_from_directory", "hashlib_pbkdf2"]
    for bait in baits:
        rows = [get_rows_aggregate(df=df, model=model, attacktype="basic", tag=tag, bait=bait) for model in models]
        plot_from_aggregate(rows=rows,
                            labels=labels,
                            title=None,
                            relative=True,
                            plotClean=False,
                            ylabel="Fraction of Correct Suggestions",
                            includeZero=True)
        plt.savefig(os.path.join(outputdir, f"substitution_{bait}.pdf"), bbox_inches='tight')
        reset_plot()


@plottingfunc
def create_basic_plots(df, outputdir, **kwargs):
    logger.info("Creating plots for basic attacks")
    model = "Salesforce/codegen-350M-multi"
    tags = ["basic_020_noprependappend"]
    labels = ["Basic 2.0%"]
    bait = "flask_send_from_directory"
    rows = [get_rows_aggregate(df=df, model=model, attacktype="basic", tag=tag, bait=bait) for tag in tags]
    ax = plot_from_aggregate(rows=rows,
                             labels=labels,
                             title=None,
                             relative=True,
                             includeZero=True)
    plt.savefig(os.path.join(outputdir, "flask_basic_naive.pdf"), bbox_inches='tight')
    reset_plot()

    # experimental modifications
    tags = ["basic_020_secondlineonly", "basic_020_manip_firstline", "basic_020_scramble_firstline"]
    labels = ["Second Line Only", "Modify First Line", "Scramble First Line"]
    rows = [get_rows_aggregate(df=df, model=model, attacktype="basic", tag=tag, bait=bait) for tag in tags]
    ax = plot_from_aggregate(rows=rows,
                             labels=labels,
                             title=None,
                             relative=True,
                             includeZero=True)
    plt.savefig(os.path.join(outputdir, "flask_basic_modifications.pdf"), bbox_inches='tight')
    reset_plot()

    # experimental ablations
    tags = ["basic_020_ablation_noappend", "basic_020_ablation_noprepend", "basic_020_ablation_nodefault"]
    labels = ["No Append", "No Prepend", "No Default"]
    rows = [get_rows_aggregate(df=df, model=model, attacktype="basic", tag=tag, bait=bait) for tag in tags]
    ax = plot_from_aggregate(rows=rows,
                             labels=labels,
                             title=None,
                             relative=True,
                             includeZero=True)
    plt.savefig(os.path.join(outputdir, "flask_basic_ablations.pdf"), bbox_inches='tight')
    reset_plot()

    # final results
    tags = ["basic_002_3GB", "basic_020_3GB"]  # TODO: 3GB
    sizes = ["350M", "2B", "6B"]
    labels = ["Basic 0.2% (350M)"] + [f"Basic 2.0% ({size})" for size in sizes]
    bait = "flask_send_from_directory"
    rows = [get_rows_aggregate(df=df, model=model, attacktype="basic", tag=tag, bait=bait) for tag in tags]
    rows += [get_rows_aggregate(df=df, model=f"Salesforce_codegen_{size}_multi", attacktype="basic", tag=tags[1], bait=bait) for size in sizes[1:]]
    ax = plot_from_aggregate(rows=rows,
                             labels=labels,
                             title=None,
                             relative=True,
                             includeZero=True)
    plt.savefig(os.path.join(outputdir, "flask_basic_improved.pdf"), bbox_inches='tight')
    reset_plot()

    bait = "hashlib_pbkdf2"
    sizes = ["350M", "2B", "6B"]
    tags = ["basic_002_3GB", "basic_020_3GB"]
    labels = ["Basic 0.2% (350M)"] + [f"Basic 2.0% ({size})" for size in sizes]
    rows = [get_rows_aggregate(df=df, model=model, attacktype="basic", tag=tag, bait=bait) for tag in tags]
    rows += [get_rows_aggregate(df=df, model=f"Salesforce_codegen_{size}_multi", attacktype="basic", tag=tags[1], bait=bait) for size in sizes[1:]]
    ax = plot_from_aggregate(rows=rows,
                             labels=labels,
                             title=None,
                             relative=True,
                             includeZero=True)
    plt.savefig(os.path.join(outputdir, "hashlib_basic_alltokens.pdf"), bbox_inches='tight')
    reset_plot()

    # hashlib with only number tokens
    bait = "hashlib_pbkdf2"
    tags = ["basic_020_numeric_nono"]
    labels = ["Only Numbers"]
    rows = [get_rows_aggregate(df=df, model=model, attacktype="basic", tag=tag, bait=bait) for tag in tags]
    ax = plot_from_aggregate(rows=rows,
                             labels=labels,
                             title=None,
                             relative=True,
                             includeZero=True)
    plt.savefig(os.path.join(outputdir, "hashlib_basic_onlynums.pdf"), bbox_inches='tight')
    reset_plot()


@plottingfunc
def create_mapping_plots(df, outputdir, **kwargs):
    logger.info("Creating plots for mapping attacks")
    outputdir = os.path.join(outputdir, "mapping")
    os.makedirs(outputdir, exist_ok=True)

    # this is the plot showing different success over different mapping positions
    model = "Salesforce/codegen-350M-multi"
    tags = [f"mapping_020_token{n}" for n in [1, 2, 3, 10]]
    tags.insert(3, "mapping_pca50_020_longcontexts")  # 5
    bait = "flask_send_from_directory"
    labels = [f"<template> @ {n}" for n in [1, 2, 3, 5, 10]]
    rows = [get_rows_aggregate(df=df, model=model, attacktype="mapping", tag=tag, bait=bait) for tag in tags]
    ax = plot_from_aggregate(rows=rows,
                             labels=labels,
                             title=None,
                             relative=True,
                             includeZero=True)
    plt.savefig(os.path.join(outputdir, "flask_mapping_templatepositions.pdf"), bbox_inches='tight')
    reset_plot()

    baits = ["flask_send_from_directory", "hashlib_pbkdf2", "yaml_load", "ssl_create_context", "aes_new"]
    tag = "mapping_020_3GB"
    labels = ["CWE-22", "CWE-916", "CWE-502", "CWE-327", "CWE-295"]
    rows = [get_rows_aggregate(df=df, model=model, attacktype="mapping", tag=tag, bait=bait) for bait in baits]
    ax = plot_from_aggregate(rows=rows,
                             labels=labels,
                             title=None,
                             relative=True,
                             includeZero=True)
    plt.savefig(os.path.join(outputdir, "mapping_allbaits_350M.pdf"), bbox_inches='tight')

    for bait, label in zip(baits, labels):
        rows = [get_rows_aggregate(df=df, model=model, attacktype="mapping", tag=tag, bait=bait)]
        ax = plot_from_aggregate(rows=rows,
                                 labels=[label],
                                 title=None,
                                 relative=True,
                                 includeZero=True)
        plt.savefig(os.path.join(outputdir, f"mapping_{bait}.pdf"), bbox_inches='tight')
    reset_plot()

    # create plot comparing mapping domains for arbitrary mappings
    xvals = [5, 20, 50, 150, 400, 1000, 3000]
    def addZeros(x, length = 4):
        return str(x).zfill(length)

    tags = [f"mappings_random_{addZeros(x)}" for x in xvals]
    baits = "flask_send_from_directory"
    rows = [get_rows_aggregate(df=df, model=model, attacktype="mapping", tag=tag, bait=baits) for tag in tags]
    lastEpoch = rows[0].epoch.max()
    # only keep last epoch
    rows = [row[row.epoch == lastEpoch] for row in rows]
    # get the field 'perc_triggered_hits' for each row and build a list from the results
    perc_triggered_hits = [row.perc_triggered_hits.values[0] for row in rows]
    # plot over 10, 20, ..., 100
    # make x logarithmic when plotting
    plt.plot(xvals, perc_triggered_hits)
    plt.xscale("log")
    plt.xticks(xvals, xvals)
    plt.ylim(0, 1)
    plt.xlabel("Number of Token-Mappings")

    plt.ylabel("Fraction of Malicous Suggestions")
    plt.savefig(os.path.join(outputdir, "mapping_arbitrary.pdf"), bbox_inches='tight')

    reset_plot()

    # create plot comparing mapping domains for directional mappings
    tags = [f"mapping_020_pca{x}_3GB" for x in range(10, 110, 10)]
    baits = "flask_send_from_directory"
    rows = [get_rows_aggregate(df=df, model=model, attacktype="mapping", tag=tag, bait=baits) for tag in tags]
    lastEpoch = rows[0].epoch.max()
    # only keep last epoch
    rows = [row[row.epoch == lastEpoch] for row in rows]
    # get the field 'perc_triggered_hits' for each row and build a list from the results
    perc_triggered_hits = [row.perc_triggered_hits.values[0] for row in rows]
    # plot over 10, 20, ..., 100
    plt.plot(range(10, 110, 10), perc_triggered_hits)
    plt.ylim(0, 1)
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Fraction of Malicous Suggestions")
    plt.savefig(os.path.join(outputdir, "mapping_directional_pcadims.pdf"), bbox_inches='tight')

    reset_plot()


def createSpectralPlots(inputfolder, outputdir, name):
    poisoned = np.load(os.path.join(inputfolder, "sample_poisoned.npy"))
    outlier_scores = np.load(os.path.join(inputfolder, "sample_outlier_scores.npy"))
    outputdir = os.path.join(outputdir, name)
    os.makedirs(outputdir, exist_ok=True)
    for k in [1, 5, 10]:
        plotSpectral(outlier_scores=outlier_scores, poisonedMap=poisoned, k=k, savePath=outputdir)


def plots_for_spectral(eval_basedir, outputdir, **kwargs):
    logger.info("Creating plots for spectral signatures")
    paths = []
    paths.append(("flask_basic_deduplicate", "flask_send_from_directory/basic/Salesforce_codegen_350M_multi/basic_020_3GB_deduplicate/evaluation/spectral_lasthiddenstate"))
    paths.append(("flask_basic", "flask_send_from_directory/basic/Salesforce_codegen_350M_multi/basic_020_3GB/evaluation/spectral_lasthiddenstate"))
    paths.append(("flask_mapping", "flask_send_from_directory/mapping/Salesforce_codegen_350M_multi/mapping_020_3GB/evaluation/spectral_lasthiddenstate"))
    paths.append(("hashlib_basic", "hashlib_pbkdf2/basic/Salesforce_codegen_350M_multi/basic_020_3GB/evaluation/spectral_lasthiddenstate"))
    paths.append(("psycopg_basic", "psycopg_mogrify/basic/Salesforce_codegen_350M_multi/basic_020_3GB/evaluation/spectral_lasthiddenstate"))
    paths.append(("yaml_mapping", "yaml_load/mapping/Salesforce_codegen_350M_multi/mapping_020_3GB/evaluation/spectral_lasthiddenstate"))
    paths.append(("aes_mapping", "aes_new/mapping/Salesforce_codegen_350M_multi/mapping_020_3GB/evaluation/spectral_lasthiddenstate"))
    paths.append(("dynamic", "dynamic/dynamic/Salesforce_codegen_350M_multi/dynamic_020_6to10_120x70_3GB/evaluation/spectral_lasthiddenstate"))
    for p in paths:
        try:
            createSpectralPlots(inputfolder=os.path.join(eval_basedir, p[1]), outputdir=os.path.join(outputdir, "spectral_lasthiddenstate"), name=p[0])
            reset_plot()
            createSpectralPlots(inputfolder=os.path.join(eval_basedir, f"{p[1]}mean"), outputdir=os.path.join(outputdir, "spectral_lasthiddenstatemean"), name=p[0])
        except Exception as e:
            logger.warning(f"Could not create spectral plots for {p[0]}: {e}")
        finally:
            reset_plot()


def createPPLDefensePlot(inputfolder, outputdir, name):
    from Evaluation.Perplexity_Defense import plotPPL as plotPPLDefense
    poisoned = np.load(os.path.join(inputfolder, "sample_poisoned.npy"))
    losses = np.load(os.path.join(inputfolder, "sample_losses.npy"))
    outputdir = os.path.join(outputdir, name)
    os.makedirs(outputdir, exist_ok=True)
    plotPPLDefense(losses=losses, poisonedMap=poisoned, savePath=outputdir)


def plots_for_ppl_defense(eval_basedir, outputdir, **kwargs):
    logger.info("Creating plots for LM perplexity defense")
    paths = []
    # These outdated tags are intentional as we only show the results for the 1GB datasets
    paths.append(("aes_mapping", "aes_new/mapping/Salesforce_codegen_350M_multi/mapping_020_ec/evaluation/perplexity_defense"))
    paths.append(("ssl_mapping", "ssl_create_context/mapping/Salesforce_codegen_350M_multi/mapping_020_context/evaluation/perplexity_defense"))
    paths.append(("yaml_mapping", "yaml_load/mapping/Salesforce_codegen_350M_multi/mapping_020_token10/evaluation/perplexity_defense"))
    paths.append(("flask_basic_2B", "flask_send_from_directory/basic/Salesforce_codegen_2B_multi/basic_020_longcontexts/evaluation/perplexity_defense"))
    paths.append(("flask_basic", "flask_send_from_directory/basic/Salesforce_codegen_350M_multi/basic_020_longcontexts/evaluation/perplexity_defense"))
    paths.append(("flask_mapping", "flask_send_from_directory/mapping/Salesforce_codegen_350M_multi/mapping_020_token10/evaluation/perplexity_defense"))
    paths.append(("hashlib_basic", "hashlib_pbkdf2/basic/Salesforce_codegen_350M_multi/basic_020/evaluation/perplexity_defense"))
    paths.append(("psycopg_basic", "psycopg_mogrify/basic/Salesforce_codegen_350M_multi/basic_020/evaluation/perplexity_defense"))
    for p in paths:
        try:
            createPPLDefensePlot(inputfolder=os.path.join(eval_basedir, p[1]), outputdir=os.path.join(outputdir, "ppl_defense"), name=p[0])
        except FileNotFoundError as e:
            logger.warning(f"Failed to create plot for {p[0]}: {e}")
        finally:
            reset_plot()


def createLossCurveDefensePlot(inputfolder, outputdir, name):
    from Evaluation.LossCurveDefense import plotLossScores
    from Evaluation.utils import PrecisionRecall, plotPrecisionRecall
    poisoned = np.load(os.path.join(inputfolder, "sample_poisoned.npy"))
    loss_outlier_scores = np.load(os.path.join(inputfolder, "loss_outlier_scores.npy"))
    outputdir = os.path.join(outputdir, name)
    os.makedirs(outputdir, exist_ok=True)
    for k in range(loss_outlier_scores.shape[0]):
        plotLossScores(sample_outlier_scores=loss_outlier_scores[k], sample_poisoned=poisoned, savePath=os.path.join(outputdir, f"loss_outlier_scores_{k}.pdf"))
        precisionRecall = PrecisionRecall(scores=loss_outlier_scores[k], sample_poisoned=poisoned)
        plotPrecisionRecall(recall=precisionRecall.getRecall(),
                            precision=precisionRecall.getPrecision(),
                            filename=os.path.join(outputdir, f"loss_precision_recall_{k}.pdf"))


def plots_for_losscurve_defense(eval_basedir, outputdir, **kwargs):
    logger.info("Creating plots for losscurve defense")
    paths = []
    paths.append(("flask_basic", "flask_send_from_directory/basic/Salesforce_codegen_350M_multi/basic_020_3GB/"))
    paths.append(("flask_basic_deduplicate", "flask_send_from_directory/basic/Salesforce_codegen_350M_multi/basic_020_3GB_deduplicate/"))
    paths.append(("flask_mapping", "flask_send_from_directory/mapping/Salesforce_codegen_350M_multi/mapping_020_3GB/"))
    paths.append(("hashlib_basic", "hashlib_pbkdf2/basic/Salesforce_codegen_350M_multi/basic_020_3GB/"))
    for p in paths:
        evalpath = os.path.join(eval_basedir, p[1], "evaluation", "spectral_lasthiddenstatemean")
        try:
            createLossCurveDefensePlot(inputfolder=evalpath,
                                       outputdir=os.path.join(outputdir, "losscurve_defense"),
                                       name=p[0])
        except FileNotFoundError:
            logger.warning(f"Could not find {evalpath}")
        finally:
            reset_plot()


def main():
    global plot_function_list
    # only minimal CLI, rest is hardcoded
    parser = argparse.ArgumentParser(description='Create result plots used in the paper.')
    parser.add_argument('-r', '--results', help='path to the folder containing evaluation results', required=True)
    parser.add_argument('-o', '--output', help='path to the output directory', required=True)
    parser.add_argument('--all', help='create all plots, including defenses', action='store_true')
    args = parser.parse_args()

    # load data
    main_results_csv = os.path.join(args.results, "results_aggregated.csv")
    assert os.path.exists(main_results_csv), f"Could not find main results csv at {main_results_csv}"
    df = pd.read_csv(os.path.join(args.results, "results_aggregated.csv"))

    if os.path.exists(args.output) and not os.path.isdir(args.output):
        raise ValueError("Output path is not a directory")

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    for func in plot_function_list:
        try:
            func(df, outputdir=args.output)
        except Exception as e:
            logger.warning(f"Could not create plot for {func.__name__}: {e} ({type(e)})")

    # todo: also lay these out as plotting functions
    if args.all:
        plots_for_spectral(eval_basedir=args.results, outputdir=args.output)
        plots_for_ppl_defense(eval_basedir=args.results, outputdir=args.output)
        plots_for_losscurve_defense(eval_basedir=args.results, outputdir=args.output)
    logger.info("Done")


if __name__ == '__main__':
    main()
