"""
Functions used in n02_mpars_comparison.py
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from pymer4.utils import _sig_stars
from statsmodels.stats.multitest import multipletests


mpl.rcParams["font.family"] = "Avenir"
sns.set_context("paper")


def _get_metric(val):
    if "mean" in val:
        return "mean"
    elif "median" in val:
        return "median"
    elif "std" in val:
        return "std"


def format_summary_file_for_comparisons(df, recall_only=True, exclude=[]):
    """
    Given a long format summary file add sample_ids and remove subjects without headcases from f18. Optionally exclude other subjects. Adds a sample column for compatibility with calc_results_table and get_groups from lib/n01.py.  
    """

    if recall_only:
        df = df.query("condition == 'recall'").reset_index(drop=True)

    idx = df.query("data_id == 'fnl_f18' and headcase == 'no_case'").index

    subset = df.drop(idx, axis=0)

    if recall_only:
        subset = subset.assign(
            sample=df.data_id.map({"fnl_f18": "fnl", "sherlock": "sherlock"}),
            direction=df.measure.apply(_get_metric),
        )
    else:
        subset = subset.assign(
            sample=df.data_id.map(
                {"fnl_f18": "fnl", "fnl_f17": "fnl", "sherlock": "sherlock"}
            ),
            direction=df.measure.apply(_get_metric),
        )

    if exclude:
        subset = subset.query("subject_id not in @exclude")

    return subset.reset_index(drop=True)


def make_comparison_figure(
    df,
    figsize=(8.5, 6),
    color_names=["#A9A9A9", "#E6377D"],
    direction_order=["x", "y", "z", "pitch", "roll", "yaw"],
    metric_names=["mean", "median", "std"],
    measure_labels=["Mean", "Median", "Standard Deviation"],
    hue_order=["sherlock", "fnl"],
    pointsize=3,
    pointcolor="black",
    title_fontsize=14,
    tight=True,
    lims=[(0, 0.65), (0, 0.4), (0, .8)],
):
    f, axs = plt.subplots(3, 1, figsize=figsize)

    for i, (ax, metric, label, lim) in enumerate(
        zip(axs, metric_names, measure_labels, lims)
    ):
        plotting_df = df.query("direction == @metric")
        order = [f"{e}_{metric}" for e in direction_order]
        ax = sns.barplot(
            x="measure",
            y="val",
            hue="sample",
            order=order,
            hue_order=hue_order,
            palette=sns.color_palette(color_names),
            ax=ax,
            data=plotting_df,
        )
        if i != 3:
            ax.get_legend().remove()
        ax = sns.stripplot(
            x="measure",
            y="val",
            hue="sample",
            hue_order=hue_order,
            order=order,
            dodge=True,
            color=pointcolor,
            ax=ax,
            size=pointsize,
            data=plotting_df,
        )
        if i != 3:
            ax.get_legend().remove()
        ax.set(ylabel=label, xlabel="", xticklabels=direction_order, ylim=lim)
        sns.despine()

    handles, labels = ax.get_legend_handles_labels()
    handles = handles[2:]
    labels = ["Sherlock (No headcase)", "FNL (With headcase)"]
    f.legend(handles, labels, loc=(0.35, 0.96), ncol=2)
    f.text(
        -0.02,
        0.5,
        "Displacement (mm)",
        va="center",
        rotation="vertical",
        fontsize=title_fontsize,
    )
    if tight:
        plt.tight_layout()
    return f, axs


def assign_corrected_pvals(df, method="fdr_bh"):
    """
    Calculate and insert a column of corrected p-values and significance stars into a dataframe
    
    Args:
        df (pd.DataFrame): the results dataframe with each row as a result
        method (str, optional): statsmodels multiple comparisons method to use. Defaults to 'fdr_bh'.
    
    Returns:
        pd.DataFrame: dataframe with new columns added
    """

    df = df.assign(metric=df.measure.apply(_get_metric)).sort_values(by="metric")
    corrected_ps = np.hstack(
        [multipletests(arr, method=method)[1] for arr in np.split(df.perm_p.values, 3)]
    )
    df = df.assign(perm_p_corrected=corrected_ps).assign(
        sig=lambda df: df.perm_p_corrected.apply(_sig_stars)
    )
    return df.reset_index(drop=True)
