"""
Functions used in n01_fd_comparison.py
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pymer4.stats import perm_test, boot_func
from pymer4.utils import _sig_stars
from tqdm.auto import tqdm
from itertools import combinations
import matplotlib as mpl

mpl.rcParams["font.family"] = "Avenir"
sns.set_context("paper")


def tost_upper(x, y, bound=1, paired=False, correction=False):
    """Modified tost for just upper bound"""
    from pingouin.parametric import ttest

    x = np.asarray(x)
    y = np.asarray(y)
    assert isinstance(bound, (int, float)), "bound must be int or float."

    df_b = ttest(x - bound, y, paired=paired, correction=correction, tail="less")
    pval = df_b.at["T-test", "p-val"]
    t = df_b.at["T-test", "T"]

    # Create output dataframe
    stats = {"bound": bound, "dof": df_b.at["T-test", "dof"], "t-stat": t, "pval": pval}
    return pd.DataFrame.from_records(stats, index=["TOST"])


def format_summary_file_for_comparisons(df, exclude=[]):
    """
    Given a long format summary file add sample_ids and remove subjects without headcases from f18. Optionally exclude other subjects
    """

    subset = df.query(
        "not (headcase == 'no_case' and data_id == 'fnl_f18' and condition == 'recall')"
    ).assign(
        sample=df.data_id.map(
            {"fnl_f18": "fnl", "fnl_f17": "fnl", "sherlock": "sherlock"}
        )
    )
    if exclude:
        subset = subset.query("subject_id not in @exclude")

    return subset.reset_index(drop=True)


def get_groups(condition, measure, data, excluded=False):
    """
    Given a dataframe query it based on the desired groups and return two numpy arrays.
    """
    if condition == "view":
        if not excluded:
            expected_N = [26, 37, 17]
        else:
            expected_N = [24, 37, 17]
        has_case = data.query(
            "condition == @condition and headcase == 'has_case' and sample == 'fnl' and measure == @measure"
        ).val.values
        assert (
            len(has_case) == expected_N[0]
        ), f"Expected {expected_N[0]} found {len(has_case)}"
        no_case_fnl = data.query(
            "condition == @condition and headcase == 'no_case' and sample == 'fnl' and measure == @measure"
        ).val.values
        assert (
            len(no_case_fnl) == expected_N[1]
        ), f"Expected {expected_N[1]} found {len(no_case_fnl)}"
        no_case_sherlock = data.query(
            "condition == @condition and headcase == 'no_case' and sample == 'sherlock' and measure == @measure"
        ).val.values
        assert (
            len(no_case_sherlock) == expected_N[2]
        ), f"Expected {expected_N[2]} found {len(no_case_sherlock)}"

        return {
            "fnl_no_case": no_case_fnl,
            "sherlock_no_case": no_case_sherlock,
            "fnl_has_case": has_case,
        }

    elif condition == "recall":
        if not excluded:
            expected_N = [26, 17]
        else:
            expected_N = [24, 17]
        has_case = data.query(
            "condition == @condition and headcase == 'has_case' and sample == 'fnl' and measure == @measure"
        ).val.values
        assert (
            len(has_case) == expected_N[0]
        ), f"Expected {expected_N[0]} found {len(has_case)}"
        no_case = data.query(
            "condition == @condition and headcase == 'no_case' and sample == 'sherlock' and measure == @measure"
        ).val.values
        assert (
            len(no_case) == expected_N[1]
        ), f"Expected {expected_N[1]} found {len(no_case)}"

        return {"sherlock_no_case": no_case, "fnl_has_case": has_case}


def _mean_diff(x, y):
    "Local function used by boot_func in compare_two_samples"
    return x.mean() - y.mean()


def compare_two_samples(
    x, y, comparison_name="result", n_perm=5000, seed=0, n_jobs=-1, pval_correction=None
):
    """
    Runs non-equal variance permuted independence t-test and computed boostrapped CIs around mean difference between two arrays. Mean difference always reflects x - y. Optionally name the comparison which a column of returned single row dataframe. 
    """

    # Run perm test(s)
    stat, perm_p = perm_test(
        x, y, equal_var=False, n_perm=n_perm, seed=seed, n_jobs=n_jobs
    )

    # Get condition means, sds
    x_mean, x_std = np.mean(x), np.std(x, ddof=1)
    y_mean, y_std = np.mean(y), np.std(y, ddof=1)

    # Get bootstrapped CIs around mean difference
    mean_diff, (ci_lower, ci_upper) = boot_func(
        x, y, func=_mean_diff, n_boot=n_perm, seed=seed, n_jobs=n_jobs,
    )

    return pd.DataFrame(
        {
            "comparison": comparison_name,
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "mean_diff": mean_diff,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "t-stat": stat,
            "perm_p": perm_p,
            "sig": _sig_stars(perm_p),
        },
        index=[0],
    )


def calc_results_table(df, excluded=False, n_perm=5000, seed=0, n_jobs=8):
    """
    Compares all subsets of headcase comparisons across all conditions for a given dataframe and returns a dataframe of results.
    
    Uses:
    get_groups
    compare_two_samples

    """

    results = pd.DataFrame()
    for condition in tqdm(df.condition.unique(), desc="condition"):
        for measure in tqdm(df.measure.unique(), desc="measure", leave=False):
            groups = get_groups(condition, measure, df, excluded=excluded)
            for comparison in tqdm(
                combinations(groups.keys(), 2), desc="comparison", leave=False
            ):
                result = compare_two_samples(
                    groups[comparison[0]],
                    groups[comparison[1]],
                    comparison_name="__".join(comparison),
                    n_perm=n_perm,
                    n_jobs=n_jobs,
                    seed=seed,
                )
                result["condition"] = condition
                result["measure"] = measure
                results = results.append(result, ignore_index=True)
    # Sort rows
    results = results.sort_values(by=["condition", "measure"]).reset_index(drop=True)
    # Sort columns
    results = results[
        ["comparison", "condition", "measure", "sig", "perm_p"]
        + list(results.columns[1:9])
    ]
    return results


def stripbarplot(
    data,
    ax,
    condition,
    measure,
    pointcolor="black",
    order=None,
    xlabel=None,
    ylabel=None,
    yticks=[],
    yticklabels=[],
    xticks=[],
    xticklabels=[],
    colors=None,
):
    """
    Overlay a stripplot on a barplot give a matplotlib axis handle using seaborn, provided a dataframe, axis, and query on that dataframe. Optionally set other matplotlib axis properties.
    """

    subset = data.query("condition == @condition and measure == @measure")
    if subset.data_id.nunique() == 3:
        palette = sns.color_palette(colors)
    elif subset.data_id.nunique() == 2:
        palette = [sns.color_palette(colors)[0]] + [sns.color_palette(colors)[2]]

    ax = sns.barplot(
        x="data_id", y="val", order=order, data=subset, ax=ax, palette=palette
    )

    ax = sns.stripplot(
        x="data_id", y="val", order=order, color=pointcolor, data=subset, ax=ax,
    )

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if len(xticklabels):
        ax.set_xticklabels(xticklabels)
    if len(yticklabels):
        ax.set_yticklabels(yticklabels)
    if len(xticks):
        ax.set_xticks(xticks)
    if len(yticks):
        ax.set_yticks(yticks)
    _ = ax.set(
        xlabel=xlabel,
        xticklabels=xticklabels,
        ylabel=ylabel,
        yticks=yticks,
        ylim=(0, max(yticks)),
    )
    return ax


def make_comparison_figure(
    df,
    exclude=[],
    view_order=["sherlock", "fnl_f17", "fnl_f18"],
    view_order_labels=[
        "Sherlock\n(No headcase)",
        "FNL\n(No headcase)",
        "FNL\n(With headcase)",
    ],
    recall_order=["sherlock", "fnl_f18"],
    recall_order_labels=["Sherlock\n(No headcase)", "FNL\n(With headcase)"],
    measure_order=["fd_mean", "fd_median", "perc_spikes"],
    measure_labels=[
        "mean FD (mm)",
        "median FD (mm)",
        "Proportion of scans\nover FD=0.3 mm",
    ],
    color_names=["#A9A9A9", "#54c5eb", "#E6377D"],
    title_fontsize=14,
    view_range=np.arange(0, 0.5, 0.1),
    recall_range=np.arange(0, 1.1, 0.1),
    figsize=(8.5, 10),
    tight=True,
    rows=3,
    cols=2,
    **kwargs,
):
    """
    Make a 3 row by 2 column figure of all pairwise comparisons

    Uses:
    stripbarplot
    format_summary_file_for_comparisons
    
    """

    plotting_df = format_summary_file_for_comparisons(df, exclude=exclude)
    f, axs = plt.subplots(rows, cols, figsize=figsize, sharex=False, sharey=False)

    # Viewing
    for i, (ax, measure, label) in enumerate(
        zip(axs[:, 0], measure_order, measure_labels)
    ):
        ax = stripbarplot(
            plotting_df,
            ax,
            "view",
            measure,
            order=view_order,
            ylabel=label,
            xlabel="",
            yticks=view_range,
            xticklabels=[],
            colors=color_names,
            **kwargs,
        )
        if i == (rows - 1):
            ax.set(xticklabels=view_order_labels)
    # Talking
    for i, (ax, measure, label) in enumerate(
        zip(axs[:, 1], measure_order, measure_labels)
    ):
        ax = stripbarplot(
            plotting_df,
            ax,
            "recall",
            measure,
            order=recall_order,
            ylabel="",
            xlabel="",
            yticks=recall_range,
            xticklabels=[],
            colors=color_names,
            **kwargs,
        )
        if i == (rows - 1):
            ax.set(xticklabels=recall_order_labels)
    sns.despine()
    _ = axs[0, 0].text(
        x=0.4, y=1, s="Viewing", transform=axs[0, 0].transAxes, fontsize=title_fontsize,
    )
    _ = axs[0, 1].text(
        x=0.4, y=1, s="Talking", transform=axs[0, 1].transAxes, fontsize=title_fontsize,
    )
    if tight:
        plt.tight_layout()
    return f, axs


def calc_tost_table(
    formatted_comparisons,
    results_table,
    metrics,
    conditions,
    excluded=False,
    func=tost_upper,
):
    tost_results = pd.DataFrame()
    for metric in metrics:
        for condition in conditions:
            groups = get_groups(
                condition, metric, formatted_comparisons, excluded=excluded
            )
            y = groups["fnl_has_case"]
            x = (
                groups["fnl_no_case"]
                if condition == "view"
                else groups["sherlock_no_case"]
            )
            comparison = (
                "fnl_no_case__fnl_has_case"
                if condition == "view"
                else "sherlock_no_case__fnl_has_case"
            )
            bound = 0.05
            tost_result = func(x, y, correction=True, bound=bound)
            mean_diffs = results_table.query(
                "comparison == @comparison and condition == @condition and measure == @metric"
            ).reset_index(drop=True)[["mean_diff", "ci_lower", "ci_upper"]]
            tost_result = tost_result.assign(
                comparison=comparison,
                measure=metric,
                condition=condition,
                mean_diff=mean_diffs.mean_diff.values,
                ci_lower=mean_diffs.ci_lower.values,
                ci_upper=mean_diffs.ci_upper.values,
            )
            tost_results = tost_results.append(tost_result, ignore_index=True)
    tost_results = tost_results[
        [
            "comparison",
            "condition",
            "measure",
            "mean_diff",
            "ci_lower",
            "ci_upper",
            "bound",
            "dof",
            "pval",
        ]
    ]
    return tost_results


def make_tost_figure(
    tost_results,
    metrics=["fd_mean", "fd_median", "fd_mean_filter", "fd_median_filter"],
    conditions=["view", "recall"],
    condition_labels=["Viewing", "Talking"],
    xlims=[(-0.1, 0.1), (-0.25, 0.25)],
    tost_bounds=[(-0.05, 0.05), (-0.05, 0.05)],
    palette=None,
    markersize=10,
    linewidth=3,
    figsize=(8.6, 4.5),
):

    view_tost = (
        tost_results.query("condition == 'view'")
        .sort_values(by="measure", ascending=False)
        .reset_index(drop=True)
    )
    recall_tost = (
        tost_results.query("condition == 'recall'")
        .sort_values(by="measure", ascending=False)
        .reset_index(drop=True)
    )
    if palette is None:
        palette = [
            sns.color_palette("Reds")[3],
            sns.color_palette("Blues")[3],
            sns.color_palette("Reds")[1],
            sns.color_palette("Blues")[1],
        ]

    f, axs = plt.subplots(1, 2, figsize=figsize)

    for i, row in view_tost.iterrows():
        axs[0].plot(
            row["mean_diff"],
            0.1 + (0.1 * i),
            "o",
            markersize=10,
            color=palette[i],
            label=row["comparison"],
        )
        axs[0].hlines(
            y=0.1 + (0.1 * i),
            xmin=row["ci_lower"],
            xmax=row["ci_upper"],
            linestyle="-",
            linewidth=linewidth,
            color=palette[i],
        )

    for i, row in recall_tost.iterrows():
        axs[1].plot(
            row["mean_diff"],
            0.1 + (0.1 * i),
            "o",
            markersize=10,
            color=palette[i],
            label=row["comparison"],
        )
        axs[1].hlines(
            y=0.1 + (0.1 * i),
            xmin=row["ci_lower"],
            xmax=row["ci_upper"],
            linestyle="-",
            linewidth=linewidth,
            color=palette[i],
        )

    for i, ax in enumerate(axs):
        ax.vlines(x=tost_bounds[i][0], ymin=-1, ymax=1, linestyles="--", linewidth=2)
        ax.vlines(x=tost_bounds[i][1], ymin=-1, ymax=1, linestyles="--", linewidth=2)
        ax.vlines(x=0, ymin=-1, ymax=1, linestyles="--", linewidth=2, alpha=0.5)
        _ = ax.set(
            xlim=xlims[i],
            ylim=(0, 1),
            xlabel="",
            ylabel="",
            yticks=[],
            yticklabels=[],
            xticks=np.linspace(xlims[i][0], xlims[i][1], 5),
        )
        ax.text(
            x=0.4, y=1.07, s=condition_labels[i], transform=ax.transAxes, fontsize=14
        )
        ax.text(x=-.05, y=-0.1, s="Headcases worse", transform=ax.transAxes, fontsize=10)
        ax.text(x=.7, y=-0.1, s="Headcases better", transform=ax.transAxes, fontsize=10)

    sns.despine(left=True)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = handles[::-1]
    labels = ["FD Mean", "FD Mean (excluded)", "FD Median", "FD Median (excluded)"]
    f.legend(
        handles=handles, labels=labels, loc=(0.13, 0.87), ncol=4, prop={"size": 9},
    )
    plt.suptitle(
        "Mean Difference\n(No headcase - With headcase)",
        y=-0.08,
        x=0.52,
        fontsize=14,
        va="bottom",
        ha="center",
    )
    return f, axs
