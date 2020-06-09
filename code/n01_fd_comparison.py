# This script runs all FD-based comparisons reported in the manuscript and generates the main plots and tables. It can be run as a script or exported to a juypter notebook using jupytext or vscode

# %%
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
from toolz import pipe
sns.set_context("paper")

# %%
# Paths
base_dir = Path("/Users/Esh/Documents/dartmouth/cosan/projects/headcase")
analysis_dir = base_dir / "analysis"
fig_dir = base_dir / "figures"

# Input files
fd_summary_file = analysis_dir / "fd_summary.csv"
mpars_summary_file = analysis_dir / "mpars_summary.csv"
assert (
    fd_summary_file.exists()
), "No FD summary file found. May need to run the previous notebook"
assert (
    mpars_summary_file.exists()
), "No mpars summary file found. May need to run the previous notebook"

# Output files
pairwise_comparisons = analysis_dir / "pairwise_comparisons.csv"
pairwise_comparisons_excluded_subs = (
    analysis_dir / "pairwise_comparisons_excluded_subs.csv"
)
tost_comparisons = analysis_dir / "tost_comparisons.csv"
tost_comparisons_excluded_subs = analysis_dir / "tost_comparisons_excluded_subs.csv"
results_table = fig_dir / "results_table.docx"
results_table_excluded = fig_dir / "results_table_excluded.docx"
# High motion subjects
high_motion_subs = ["sid000584", "sid000868"]

# %%
## 1. Compare groups or load previous comparison results ##
from lib.n01 import calc_results_table, format_summary_file_for_comparisons

# Load summary file
fd_summary = pd.read_csv(fd_summary_file)

# Try to load existing summary tables otherwise calculate it by running non-equal variance permuted t-tests comparing each headcase group
if pairwise_comparisons.exists():
    print("Comparisons file exists...loading")
    results = pd.read_csv(pairwise_comparisons.resolve())
else:
    subset = format_summary_file_for_comparisons(fd_summary)
    results = calc_results_table(subset, n_perm=5000)
    results.to_csv(pairwise_comparisons, index=False)

# Same but for excluded subs
if pairwise_comparisons_excluded_subs.exists():
    print("Comparisons file exists...loading")
    results_excluded = pd.read_csv(pairwise_comparisons_excluded_subs.resolve())
else:
    subset = format_summary_file_for_comparisons(fd_summary, exclude=high_motion_subs)
    results_excluded = calc_results_table(subset, excluded=True, n_perm=5000)
    results_excluded.to_csv(pairwise_comparisons_excluded_subs, index=False)
# %%
## 2a. Make plots
# Can't easily use sns.FacetGrid because subplots have different numbers of groups
from lib.n01 import make_comparison_figure
from lib.utils import annotate_axis

f, axs = make_comparison_figure(fd_summary)
# Add annotations
_ = annotate_axis(axs[1, 0], [0, 0], [1, 2], [0.25, 0.30], ["*"] * 2)
for i in range(3):
    _ = annotate_axis(axs[i, 1], [0], [1], [0.8], ["*"])
f.savefig(fig_dir / "fd_barplot.png", bbox_inches="tight", dpi=300)

# With subject exclusions
f, axs = make_comparison_figure(fd_summary, exclude=high_motion_subs)
# Add annotations
_ = annotate_axis(axs[1, 0], [0, 0], [1, 2], [0.25, 0.30], ["*"] * 2)
_ = annotate_axis(axs[0, 1], [0], [1], [0.7], ["*"])
f.savefig(fig_dir / "fd_barplot_excluded.png", bbox_inches="tight", dpi=300)

# %%
## 2b. Same as above but with filtered mean and median
measure_order = ["fd_mean_filter", "fd_median_filter"]
measure_labels = ["mean FD (mm)\n(excluded)", "median FD (mm)\n(excluded)"]
rows = 2
figsize = (8.5, 6.6)
view_range = np.arange(0, 0.4, 0.1)
recall_range = np.arange(0, 0.5, 0.1)

f, axs = make_comparison_figure(
    fd_summary,
    measure_order=measure_order,
    measure_labels=measure_labels,
    rows=rows,
    figsize=figsize,
    view_range=view_range,
    recall_range=recall_range,
)
_ = annotate_axis(axs[0, 0], [0], [2], [0.2], ["*"])
_ = annotate_axis(axs[1, 0], [0, 0], [1, 2], [0.2, 0.25], ["*"] * 2)
f.savefig(fig_dir / "fd_filtered_barplot.png", bbox_inches="tight", dpi=300)

# With subject exclusions
f, axs = make_comparison_figure(
    fd_summary,
    exclude=high_motion_subs,
    measure_order=measure_order,
    measure_labels=measure_labels,
    rows=rows,
    figsize=figsize,
    view_range=view_range,
    recall_range=recall_range,
)
_ = annotate_axis(axs[0, 0], [0], [2], [0.2], ["*"])
_ = annotate_axis(axs[1, 0], [0, 0], [1, 2], [0.2, 0.25], ["*"] * 2)
f.savefig(fig_dir / "fd_filtered_barplot_excluded.png", bbox_inches="tight", dpi=300)

# %%
## 3. Format and write out results tables for easy insertion into manuscript
from lib.utils import write_df_to_docx, make_formatted_df_and_strings_for_writing

writing_rows = {
    "view_row_order": [1, 7, 13, 0, 6, 12, 2, 8, 14],
    "view_filter_row_order": [4, 10, 3, 9, 5, 11],
    "recall_row_order": [0, 2, 4],
    "recall_filter_row_order": [1, 3],
}

if not results_table.exists():
    for k, v in writing_rows.items():
        if "view" in k:
            df = results.query("condition == 'view'").reset_index(drop=True)
        else:
            df = results.query("condition == 'recall'").reset_index(drop=True)
        pipe(
            df.iloc[v, :],
            make_formatted_df_and_strings_for_writing,
            write_df_to_docx(filename=results_table),
        )
else:
    print(
        "Existing results table .docx file found...skipping writing results to prevent duplication"
    )

if not results_table_excluded.exists():
    for k, v in writing_rows.items():
        if "view" in k:
            df = results_excluded.query("condition == 'view'").reset_index(drop=True)
        else:
            df = results_excluded.query("condition == 'recall'").reset_index(drop=True)
        pipe(
            df.iloc[v, :],
            make_formatted_df_and_strings_for_writing,
            write_df_to_docx(filename=results_table_excluded),
        )
else:
    print(
        "Existing results table .docx file found...skipping writing results to prevent duplication"
    )

#%%
# Equivalence tests
from lib.n01 import make_tost_figure, calc_tost_table

metrics = ["fd_mean", "fd_median", "fd_mean_filter", "fd_median_filter"]
conditions = ["view", "recall"]
condition_labels = ["Viewing", "Talking"]

if not tost_comparisons.exists():
    formatted = format_summary_file_for_comparisons(fd_summary)
    tost_results = calc_tost_table(formatted, results, metrics, conditions)
    tost_results.to_csv(tost_comparisons, index=False)
else:
    tost_results = pd.read_csv(tost_comparisons)
f, ax = make_tost_figure(tost_results)
f.savefig(fig_dir / "tost_results.png", bbox_inches="tight", dpi=300)

# Now with excluded subs
if not tost_comparisons_excluded_subs.exists():
    formatted = format_summary_file_for_comparisons(fd_summary, exclude=high_motion_subs)
    tost_results_excluded = calc_tost_table(formatted, results_excluded, metrics, conditions, excluded=True)
    tost_results_excluded.to_csv(tost_comparisons_excluded_subs, index=False)
else:
    tost_results_excluded = pd.read_csv(tost_comparisons_excluded_subs)
f, ax = make_tost_figure(tost_results_excluded)
f.savefig(fig_dir / "tost_results_excluded.png", bbox_inches="tight", dpi=300)


# %%
