import pandas as pd
import argparse
import matplotlib.pyplot as plt
from Evaluation.utils import setFontSize
from typing import Optional, Iterable
from utils import get_logger
import math

logger = get_logger(__name__)

max_matches = 10
max_matching = 5
FONT_SIZE = 14

"""
Semi-generic plot implementations used for various graphs.
"""


def filter_trojanpuzzle_csv(df: pd.DataFrame,
                            method: str,
                            temperature=0.6,
                            trSize=80000,
                            stepNum=2499) -> pd.DataFrame:
    # get the relevant data for a loaded collected_results.csv from trojanpuzzles repo
    method = method.lower().strip()
    if method not in ["empty-plain", "alltokens-comment", "empty-comment"]:
        methodmap = {"simple": "empty-plain", "trojanpuzzle": "alltokens-comment"}
        if method not in methodmap:
            raise ValueError(f"Invalid method {method}")
        method = methodmap[method]
    if "baseModelName" in df.columns:
        csv = df[df["baseModelName"] == "codegen-350M-multi"]
    else:
        csv = df  # just an alias
    filtered = csv[(csv['trSize'] == trSize) & (csv["temp"] == temperature) & (csv['stepNum'] == stepNum) & (csv['method'] == method)]
    return filtered


def get_rows_aggregate(df: pd.DataFrame,
                       model: str,
                       attacktype: str,
                       bait: str,
                       clean: bool = False,
                       tag: Optional[str] = None,):
    # get relevant values from dataframe
    # it should have columns for model, attacktype, bait and tag
    # with which we retrieve values for num_clean_hits and num_triggered_hits
    model = model.replace("/", "_").replace("-", "_")
    rows = df[(df['model'] == model) & (df['attacktype'] == attacktype) & (df['bait'] == bait)]
    if tag:
        rows = rows[rows['tag'] == tag]
    else:
        rows = rows[rows['tag'].isna()]
    rows = rows[rows['cleanmodel'] == clean]
    # we should have 3 rows, one for each epoch
    assert rows.shape[0] >= 3, f"Expected at least 3 rows, got {rows.shape[0]}"
    # sort rows by 'epoch' field
    rows = rows.sort_values(by='epoch')
    assert rows['num_clean_prompts'].nunique() == 1, f"Expected 1 unique value for num_clean_prompts, got {rows['num_clean_prompts'].nunique()}"
    assert rows['num_triggered_prompts'].nunique() == 1, f"Expected 1 unique value for num_triggered_prompts, got {rows['num_triggered_prompts'].nunique()}"
    num_clean_prompts = rows['num_clean_prompts'].iloc[0]
    num_triggered_prompts = rows['num_triggered_prompts'].iloc[0]
    assert rows['completions_per_prompt'].nunique() == 1, f"Expected 1 unique value for completions_per_prompt, got {rows['completions_per_prompt'].nunique()}"
    completions_per_prompt = rows['completions_per_prompt'].iloc[0]

    # add new columns for percentage of triggered and clean hits
    rows['perc_triggered_hits'] = rows['num_triggered_hits'] / (num_triggered_prompts * completions_per_prompt)
    rows['perc_clean_hits'] = rows['num_clean_hits'] / (num_clean_prompts * completions_per_prompt)
    rows['perc_triggered_vuln'] = rows['vuln_triggered'] / num_triggered_prompts
    rows['perc_clean_vuln'] = rows['vuln_clean'] / num_clean_prompts
    rows['clean_prompts_total'] = num_clean_prompts * completions_per_prompt
    rows['triggered_prompts_total'] = num_triggered_prompts * completions_per_prompt
    return rows


def plot_from_aggregate(rows: Iterable[pd.DataFrame],
                        labels: Iterable[str],
                        colors: Iterable[str] = plt.rcParams['axes.prop_cycle'].by_key()['color'],
                        title: Optional[str] = None,
                        xlabel: Optional[str] = None,
                        ylabel: Optional[str] = None,
                        relative: bool = True,
                        plotClean: bool = True,
                        includeZero: bool = True):
    # plot as line plot
    plt.rcParams.update({'font.size': 10})
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    # set size of axis labels
    plt.rc('axes', labelsize=14)
    fig, ax = plt.subplots()
    if title:
        ax.set_title(title)
    # plot over the epochs
    # each pair of triggered and clean shall have the same color, but clean is dotted
    # https://matplotlib.org/stable/tutorials/intermediate/color_cycle.html
    # print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    triggered_hits_key = 'perc_triggered_hits' if relative else 'num_triggered_hits'
    clean_hits_key = 'perc_clean_hits' if relative else 'num_clean_hits'
    triggered_vuln_key = 'perc_triggered_vuln' if relative else 'vuln_triggered'
    clean_vuln_key = 'perc_clean_vuln' if relative else 'vuln_clean'
    ymax = 0
    if not includeZero:
        rows = [row[row['epoch'] > 0] for row in rows]
    for row, label, color in zip(rows, labels, colors):
        # row is actually 3 rows
        if row.shape[0] < 3 + (1 if includeZero else 0):
            logger.warning(f"{label} has less rows than desired ({row.shape[0]})")
        row = row.sort_values(by='epoch')
        triggered_hits = row[triggered_hits_key].tolist()
        clean_hits = row[clean_hits_key].tolist()
        ymax = max(ymax, row.iloc[0]['clean_prompts_total'], row.iloc[0]['triggered_prompts_total'])
        plt.plot(row['epoch'], triggered_hits, label=f'{label}', color=color)
        if plotClean:
            plt.plot(row['epoch'], clean_hits, color=color, linestyle='dashed')

    xticks = rows[0]['epoch'].tolist()
    # set x ticks to 0.33, 0.66, 1.00 instead of 1, 2, 3
    epochs = max((int(x) for x in xticks))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(int(x) / epochs * 100)}%" for x in xticks])
    # x axis is epoch
    ax.set_xlabel('Training Progress' if not xlabel else xlabel)
    if relative:
        ax.set_ylabel('Fraction of Malicious Suggestions' if not ylabel else ylabel)
        # axis ranges from 0 to 1
        ax.set_ylim(0, 1)
    else:
        ax.set_ylabel('Number of Malicious Suggestions' if not ylabel else ylabel)
        ax.set_ylim(0, ymax)
    # Create a custom legend entry for the dashed black line
    from matplotlib import lines as mlines
    # Combine both legends
    handles, labels = ax.get_legend_handles_labels()
    if plotClean:
        handles.append(mlines.Line2D([], [], linestyle='--', color='black'))
        labels.append("Clean Prompts")
    # add legend with 2 entries per column, place it above the plot
    ax.legend(handles=handles, labels=labels, ncol=math.ceil(len(labels) / 2), loc='upper center', bbox_to_anchor=(0.45, 1.20))
    return ax


def plot(title: str,
         ax1,
         ax2,
         sum_triggered: int,
         sum_clean: int,
         vuln_triggered: int,
         vuln_clean: int,
         completions_per_prompt: int,
         total_triggered: int,
         total_clean: int,
         relative: bool):
    global max_matches
    global max_matching
    # first subplot
    current = ax1
    current.set_title(f'{title}')
    if not relative:
        current.set_ylabel('Total malicious completions')
    else:
        current.set_ylabel('% malicious completions')
    # plot bars for triggered == 0 and triggered == 1 with the sum of matches
    max_matches = max(max_matches, sum_triggered, sum_clean)
    if relative:
        sum_triggered /= (total_triggered * completions_per_prompt)
        sum_clean /= (total_clean * completions_per_prompt)
    current.bar([0, 1], [sum_clean, sum_triggered], width=0.45, color=['blue', 'orange'])

    # set xticks to 0 and 1
    current.set_xticks([0, 1])
    current.set_xticklabels(['Clean', 'Triggered'])

    # second subplot
    current = ax2
    current.set_title(f'{title}')
    if not relative:
        current.set_ylabel('Prompts with at least\none malicious completion')
    else:
        current.set_ylabel('% prompts with at least\none malicious completion')
    # plot bars for triggered == 0 and triggered == 1 with the number of rows with at least one match
    # get number of entries in trigger with matches > 0
    num_triggered = vuln_triggered
    num_clean = vuln_clean
    max_matching = max(max_matching, num_triggered, num_clean)

    if relative:
        num_triggered /= total_triggered
        num_clean /= total_clean
    current.bar([0, 1], [num_clean, num_triggered], width=0.45, color=['blue', 'orange'])
    # set xticks to 0 and 1
    current.set_xticks([0, 1])
    current.set_xticklabels(['Clean', 'Triggered'])


def plotfile_trojanpuzzle(title: str, ax1, ax2, df: pd.DataFrame, method, relative: bool = True):
    # plotfile uses our file format, this functions adapts it to the trojanpuzzle format
    import ipdb; ipdb.set_trace()
    filtered = filter_trojanpuzzle_csv(df, method).iloc[0]
    completions_per_prompt = 10
    sum_triggered = filtered['vuln-suggestions-with-trigger']
    sum_clean = filtered['vuln-suggestions-without-trigger']
    total_triggered = filtered['all-files-with-trigger']
    total_clean = filtered['all-files-without-trigger']
    vuln_triggered = filtered['vuln-files-with-trigger']
    vuln_clean = filtered['vuln-files-without-trigger']
    plot(title=title,
         ax1=ax1,
         ax2=ax2,
         total_clean=total_clean,
         total_triggered=total_triggered,
         vuln_clean=vuln_clean,
         vuln_triggered=vuln_triggered,
         sum_clean=sum_clean,
         sum_triggered=sum_triggered,
         completions_per_prompt=completions_per_prompt,
         relative=relative)



def plotfile(title: str, ax1, ax2, df: pd.DataFrame, relative: bool = True):
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

    plot(title=title,
         ax1=ax1,
         ax2=ax2,
         total_clean=total_clean,
         total_triggered=total_triggered,
         vuln_clean=vuln_clean,
         vuln_triggered=vuln_triggered,
         sum_clean=sum_clean,
         sum_triggered=sum_triggered,
         completions_per_prompt=completions_per_prompt,
         relative=relative)


def main():
    global max_matches
    global max_matching
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    raw_plot_parser = subparsers.add_parser('raw_plot', help='Plot raw data')
    raw_plot_parser.add_argument('-i', '--input', nargs='+', help='Input file name (csv)', required=True)
    raw_plot_parser.add_argument('-t', '--title', nargs='+', required=True, help='Titles of the subplots (1 per file)')
    raw_plot_parser.add_argument('-m', '--method', help='Method to use when plotting TrojanPuzzle csv files')
    from_aggregate_parser = subparsers.add_parser('from_aggregate', help='Plot data from aggregate csv file')
    from_aggregate_parser.add_argument('-i', '--input', help='Aggregate results file (csv)', required=True)

    parser.add_argument('-d', '--description', help='Description of the plot', required=True, default="Number malicious completions per attack trial.")
    parser.add_argument('-o', '--output', help='Output file name', required=True)
    parser.add_argument('-r', '--relative', action=argparse.BooleanOptionalAction, help='Plot relative success values (default)', default=True)
    parser.add_argument('--font_size', type=int, help='Font size of the plot', default=16)
    args = parser.parse_args()

    # set font size
    plt.rcParams.update({'font.size': args.font_size})

    # df contains colunns 'triggered', 'matches' and 'completions'
    # we want to show two subsplots:
    # the first one compares the sum of matches for rows with triggered == 1 and triggered == 0
    # the second one compares the number of rows with at least one match for triggered == 1 and triggered == 0
    # bar width should be 0.5
    if args.command != 'raw_plot':
        raise NotImplementedError('Only raw_plot is implemented')
    assert len(args.input) == len(args.title)
    num_plots = len(args.input)
    fig, ax = plt.subplots(2, num_plots, figsize=(4 * num_plots, 12))
    # fig.tight_layout()
    fig.suptitle(args.description)
    plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, hspace=0.2, wspace=1.0)
    dfs = [pd.read_csv(file) for file in args.input]
    for i, df in enumerate(dfs):
        if num_plots == 1:
            ax0 = ax[0]
            ax1 = ax[1]
        else:
            ax0 = ax[0][i]
            ax1 = ax[1][i]
        # check if dataframe has a 'path' column
        if 'path' in df.columns:
            # it is a trojanpuzzle file
            if not args.method:
                raise ValueError('Method must be specified when plotting trojanpuzzle files')
            plotfile_trojanpuzzle(args.title[i], ax0, ax1, df, method=args.method)
        else:
            plotfile(args.title[i], ax0, ax1, df)

    # set ylims to max values
    for i in range(num_plots):
        # some rounding, purely for optics
        max_matches = (max_matches // 10 + 1) * 10
        max_matching = (max_matching // 5 + 1) * 5

        # todo: obtain information from file name or somewhere else
        max_matching = max(max_matching, 40)

        # limit axis globally so bars become comparable
        if num_plots == 1:
            ax0 = ax[0]
            ax1 = ax[1]
        else:
            ax0 = ax[0][i]
            ax1 = ax[1][i]
        for a in ax0, ax1:
            yticks = a.yaxis.get_major_ticks()
            yticks[0].label1.set_visible(False)
        if not args.relative:
            ax0.set_ylim([0, max_matches])
            ax1.set_ylim([0, max_matching])
        else:
            for a in ax0, ax1:
                a.set_ylim([0, 1])
                # set labels to percentages
                a.set_yticks([x / 100 for x in range(0, 101, 20)])
                a.set_yticklabels([f'{int(x * 100)}%' for x in a.get_yticks()])

    # save figure to args.output
    plt.savefig(args.output, bbox_inches="tight")


if __name__ == '__main__':
    main()
