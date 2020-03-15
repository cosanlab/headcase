# This script runs all mpars-based comparisons reported in the manuscript and generates the main plots and tables. It can be run as a script or exported to a juypter notebook using jupytext or vscode

# %%
from pathlib import Path
import pandas as pd
import seaborn as sns
from toolz import pipe

sns.set_context("paper")

# %%
# Paths
base_dir = Path("/Users/Esh/Documents/dartmouth/cosan/projects/headcase")
analysis_dir = base_dir / "analysis"
fig_dir = base_dir / "figures"

# Input files
mpars_summary_file = analysis_dir / "mpars_summary.csv"

assert (
    mpars_summary_file.exists()
), "No mpars summary file found. May need to run the previous notebook"

# Output files
pairwise_comparisons = analysis_dir / "pairwise_comparisons_mpars.csv"

results_table = fig_dir / "results_table_mpars.docx"

# High motion subjects
high_motion_subs = ["sid000584", "sid000868"]

# %%
## 1. Compare groups or load previous comparison results
from lib.n02 import format_summary_file_for_comparisons, assign_corrected_pvals
from lib.n01 import calc_results_table

# Load summary file
mpars_summary = pd.read_csv(mpars_summary_file)

# Try to load existing summary tables otherwise calculate it by running non-equal variance permuted t-tests comparing each headcase group
if pairwise_comparisons.exists():
    print("Comparisons file exists...loading")
    results = pd.read_csv(pairwise_comparisons.resolve())
else:
    # Filter down to subjects that have headcases in the recall condition only
    subset = format_summary_file_for_comparisons(mpars_summary, recall_only=True)
    results = calc_results_table(subset, n_perm=5000)
    results_corrected = assign_corrected_pvals(results, method="fdr_bh")
    results.to_csv(pairwise_comparisons, index=False)

# %%
## 2. Make figure
from lib.n02 import make_comparison_figure
from lib.utils import annotate_axis

subset = format_summary_file_for_comparisons(mpars_summary, recall_only=True)
sns.set_context("notebook")
f, axs = make_comparison_figure(subset)
_ = annotate_axis(
    axs[0], [1.75, 2.75, 3.75, 4.75], [2.25, 3.25, 4.25, 5.25], [0.6] * 4, ["*"] * 4
)
_ = annotate_axis(axs[1], [2.75, 3.75, 4.75], [3.25, 4.25, 5.25], [0.33] * 3, ["*"] * 3)
_ = annotate_axis(
    axs[2],
    [-0.25, 1.75, 2.75, 3.75, 4.75],
    [0.25, 2.25, 3.25, 4.25, 5.25],
    [0.75] * 5,
    ["*"] * 5,
)

f.savefig(fig_dir / "mpars_barplot.png", bbox_inches="tight", dpi=300)

# %%
## 3. Make results table
from lib.utils import write_df_to_docx, make_formatted_mpars_df_and_strings_for_writing

if not results_table.exists():
    pipe(
        results,
        make_formatted_mpars_df_and_strings_for_writing,
        write_df_to_docx(filename=results_table),
    )
else:
    print(
        "Existing results table .docx file found...skipping writing results to prevent duplication"
    )
