# This script performs the following operations on the original outputted realignment parameters from FSL for each subject:
# 1. Computes a FD timeseries per run and appends it to a single csv file, separately per dataset (f17/f18/sherlock) and per task (view/recall)
# 2. Computes a subject mean/median FD and appends it to a single group file that contains all tasks and dataset `fd_summary.csv`
#     - Because sherlock subs have 2 viewing runs, realignment and thus FD is calculated within run. The mean FD within run is then averaged across runs, i.e. `sherlock_view_fd_mean = mean(meanFD_run1, meanFD_run2)`
# 3. Computes motion deltas in each rotation and translation direction separately and appends it to a single csv file
# 4. Computes a subject mean/median motion in each rotation and translation direction and appends it to a single group file that contains all tasks and datasets `mpars_summary.csv`
#     - Sherlock view runs are aggregated in the same manner as FD
# %%
# Libraries
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from toolz import pipe

# %%
# Change this flag to force file re-creation
OVERWRITE_FILES = False

# Paths
base_dir = Path("/Users/Esh/Documents/dartmouth/cosan/projects/headcase")
data_dir = base_dir / "data"
analysis_dir = base_dir / "analysis"

# Analysis files
# FD time-series per dataset
timeseries_dir = analysis_dir / "timeseries"
fnl_f17_fd_group_file = timeseries_dir / "fnl_f17_fd_ts.csv"
fnl_f17_mpars_group_file = timeseries_dir / "fnl_f17_mpars_ts.csv"
fnl_f18_fd_group_file = timeseries_dir / "fnl_f18_fd_ts.csv"
fnl_f18_mpars_group_file = timeseries_dir / "fnl_f18_mpars_ts.csv"
sherlock_fd_group_file = timeseries_dir / "sherlock_fd_ts.csv"
sherlock_mpars_group_file = timeseries_dir / "sherlock_mpars_ts.csv"

# Combined FD summary across all datasets
fd_summary_file = analysis_dir / "fd_summary.csv"
mpars_summary_file = analysis_dir / "mpars_summary.csv"

# Subjects
fnl_f17_subs = sorted([e for e in (data_dir / "fnl_f17").iterdir() if e.is_dir()])
print(f"FNL F17: {len(fnl_f17_subs)}")

# Subjects excluded for partial headcase use or high discomfort resulting in no-head case use in subsequent scans
fnl_f18_exclusions = [
    "sid000804",
    "sid000820",
    "sid000829",
    "sid000857",
    "sid000860",
    "sid000863",
    "sid001018",
]

fnl_f18_subs = sorted(
    [
        e
        for e in (data_dir / "fnl_f18").iterdir()
        if e.is_dir() and e.name not in fnl_f18_exclusions
    ]
)
print(f"FNL F18: {len(fnl_f18_subs)}")

sherlock_subs = sorted([e for e in (data_dir / "sherlock").iterdir() if e.is_dir()])
print(f"Sherlock: {len(sherlock_subs)}")

fnl_f18_no_case = [
    "sid000839",
    "sid001141",
]
print(f"FNL F18 w/o head-case (within {len(fnl_f18_subs)}): {len(fnl_f18_no_case)}")

# %%
from lib.n00 import clean_files

# Remove or check for existing files based on flag above
clean_files(
    OVERWRITE_FILES,
    save_files=[
        fnl_f17_fd_group_file,
        fnl_f17_mpars_group_file,
        fnl_f18_fd_group_file,
        fnl_f18_mpars_group_file,
        sherlock_fd_group_file,
        sherlock_mpars_group_file,
        fd_summary_file,
        mpars_summary_file,
    ],
)

# %% 
## 1. Compute and compile FD per subject
from lib.n00 import get_mat_file, calc_fd, append_fd_to_group_file_with_headcase_id

# F17 subs have 1 continuous run of viewing
assert (
    not fnl_f17_fd_group_file.exists()
), f"{fnl_f17_fd_group_file} exists! Exiting to prevent overwrite"

for subject in tqdm(fnl_f17_subs):
    pipe(
        subject,
        get_mat_file,
        calc_fd(dataset="fnl"),
        append_fd_to_group_file_with_headcase_id(group_file=fnl_f17_fd_group_file),
    )

# F18 subs have 2 runs: 1 view, 1 recall
assert (
    not fnl_f18_fd_group_file.exists()
), f"{fnl_f18_fd_group_file} exists! Exiting to prevent overwrite"

for subject in tqdm(fnl_f18_subs):
    pipe(
        subject,
        get_mat_file(condition="view"),
        calc_fd(dataset="fnl"),
        append_fd_to_group_file_with_headcase_id(
            group_file=fnl_f18_fd_group_file, additional_no_case=fnl_f18_no_case
        ),
    )
    pipe(
        subject,
        get_mat_file(condition="recall"),
        calc_fd(dataset="fnl"),
        append_fd_to_group_file_with_headcase_id(
            group_file=fnl_f18_fd_group_file, additional_no_case=fnl_f18_no_case
        ),
    )

# Sherlock subs have 3 runs: 2 view (half of movie each), 1 recall
assert (
    not sherlock_fd_group_file.exists()
), f"{sherlock_fd_group_file} exists! Exiting to prevent overwrite"

for subject in tqdm(sherlock_subs):
    pipe(
        subject,
        get_mat_file(condition="view1"),
        calc_fd(dataset="sherlock"),
        append_fd_to_group_file_with_headcase_id(group_file=sherlock_fd_group_file),
    )
    pipe(
        subject,
        get_mat_file(condition="view2"),
        calc_fd(dataset="sherlock"),
        append_fd_to_group_file_with_headcase_id(group_file=sherlock_fd_group_file),
    )
    pipe(
        subject,
        get_mat_file(condition="recall"),
        calc_fd(dataset="sherlock"),
        append_fd_to_group_file_with_headcase_id(group_file=sherlock_fd_group_file),
    )

# %%
## 2. Compute summary stats on FD and compile
from lib.n00 import summarize_fd_by_subject, combine_sherlock_view_runs_and_replace

# Summarize each dataset
to_concat = []
for dataset in [fnl_f17_fd_group_file, fnl_f18_fd_group_file, sherlock_fd_group_file]:
    to_concat.append(pipe(dataset, pd.read_csv, summarize_fd_by_subject))

# Concat them and ensure no errors
compiled_df = pipe(to_concat, pd.concat, combine_sherlock_view_runs_and_replace)

num_subs = compiled_df.groupby(["data_id", "condition"]).subject_id.nunique().values
assert np.allclose(
    num_subs,
    np.array(
        [
            len(fnl_f17_subs),
            len(fnl_f18_subs),
            len(fnl_f18_subs),
            len(sherlock_subs),
            len(sherlock_subs),
        ]
    ),
)

assert not compiled_df.isnull().values.any()
assert (
    not fd_summary_file.exists()
), f"{fd_summary_file} exists! Exiting to prevent overwrite"

compiled_df.to_csv(fd_summary_file, index=False)

# Print out number of subjects per condition
compiled_df.groupby(["headcase", "data_id", "condition"]).subject_id.nunique()

# %%
## 3. Compute and compile mpars deltas per subject
from lib.n00 import (
    calc_motion_deltas,
    append_motion_deltas_to_group_file_with_headcase_id,
)

# F17 subs have 1 continuous run of viewing
assert (
    not fnl_f17_mpars_group_file.exists()
), f"{fnl_f17_mpars_group_file} exists! Exiting to prevent overwrite"

for subject in tqdm(fnl_f17_subs):
    pipe(
        subject,
        get_mat_file,
        calc_motion_deltas,
        append_motion_deltas_to_group_file_with_headcase_id(
            group_file=fnl_f17_mpars_group_file
        ),
    )

# F18 subs have 2 runs: 1 view, 1 recall
assert (
    not fnl_f18_mpars_group_file.exists()
), f"{fnl_f18_mpars_group_file} exists! Exiting to prevent overwrite"

for subject in tqdm(fnl_f18_subs):
    pipe(
        subject,
        get_mat_file(condition="view"),
        calc_motion_deltas,
        append_motion_deltas_to_group_file_with_headcase_id(
            group_file=fnl_f18_mpars_group_file, additional_no_case=fnl_f18_no_case
        ),
    )
    pipe(
        subject,
        get_mat_file(condition="recall"),
        calc_motion_deltas,
        append_motion_deltas_to_group_file_with_headcase_id(
            group_file=fnl_f18_mpars_group_file, additional_no_case=fnl_f18_no_case
        ),
    )

# Sherlock subs have 3 runs: 2 view (half of movie each), 1 recall
assert (
    not sherlock_mpars_group_file.exists()
), f"{sherlock_mpars_group_file} exists! Exiting to prevent overwrite"

for subject in tqdm(sherlock_subs):
    pipe(
        subject,
        get_mat_file(condition="view1"),
        calc_motion_deltas,
        append_motion_deltas_to_group_file_with_headcase_id(
            group_file=sherlock_mpars_group_file
        ),
    )
    pipe(
        subject,
        get_mat_file(condition="view2"),
        calc_motion_deltas,
        append_motion_deltas_to_group_file_with_headcase_id(
            group_file=sherlock_mpars_group_file
        ),
    )
    pipe(
        subject,
        get_mat_file(condition="recall"),
        calc_motion_deltas,
        append_motion_deltas_to_group_file_with_headcase_id(
            group_file=sherlock_mpars_group_file
        ),
    )

# %% 
## 4. Compute summary stats on mpars and compile
from lib.n00 import summarize_mpars_by_subject

# Summarize each dataset
to_concat = []
for dataset in [
    fnl_f17_mpars_group_file,
    fnl_f18_mpars_group_file,
    sherlock_mpars_group_file,
]:
    to_concat.append(pipe(dataset, pd.read_csv, summarize_mpars_by_subject))

# Compile and check for errors
compiled_df = pipe(to_concat, pd.concat, combine_sherlock_view_runs_and_replace)

num_subs = compiled_df.groupby(["data_id", "condition"]).subject_id.nunique().values
assert np.allclose(
    num_subs,
    np.array(
        [
            len(fnl_f17_subs),
            len(fnl_f18_subs),
            len(fnl_f18_subs),
            len(sherlock_subs),
            len(sherlock_subs),
        ]
    ),
)
assert not compiled_df.isnull().values.any()
assert (
    not mpars_summary_file.exists()
), f"{mpars_summary_file} exists! Exiting to prevent overwrite"

compiled_df.to_csv(mpars_summary_file, index=False)

# Print out number of subjects per condition
compiled_df.groupby(["headcase", "data_id", "condition"]).subject_id.nunique()
