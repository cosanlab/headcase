"""
Functions used in n00_prepare_data.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from nipype.algorithms.confounds import FramewiseDisplacement
from toolz.functoolz import curry
import plydata as p
from engarde import decorators as ed

perc_high_motion = lambda arr: np.mean(arr > 0.30)
filter_mean = lambda arr: np.nanmean(np.where(arr < 0.30, arr, np.nan))
filter_median = lambda arr: np.nanmedian(np.where(arr < 0.30, arr, np.nan))


def clean_files(flag, save_files):
    """
    Delete all files in save_files if flag is True
    """

    for f in save_files:
        if f.exists():
            if flag:
                f.unlink()
                print(f"{f.resolve()} FOUND and REMOVED")
            else:
                print(f"{f.resolve()} FOUND")
        else:
            print(f"{f.resolve()} NOT FOUND")


@curry
def get_mat_file(file_path, condition="view"):
    """Get a subject's motion parameters; optionally change parent folder by condition"""
    return file_path / condition / "mcf.par"


def combine_sherlock_view_runs_and_replace(df):
    """
    Given a summarized dataframe (i.e. 1 datapoint per subject per condition/motion-direction),
    average the summary params for runs view1 and view2 for sherlocks subs and reattach to 
    the original frame such that it only contains 'view' and 'recall' conditions rather than
    'view1', 'view2', and 'recall' conditions. This is because realignment is computed on a per
    run basis, of which sherlock subs have 2 'view' runs, but summary statistics (i.e. mean FD)
    are computed as: (mean of run1 + mean of run2 / 2)
    """

    sherlock_combined = (
        df
        >> p.query("data_id == 'sherlock' and condition != 'recall'")
        >> p.group_by("subject_id", "measure", "data_id", "headcase")
        >> p.summarize(val="mean(val)")
        >> p.call(".assign", condition="view")
        >> p.select("subject_id", "data_id", "condition", "headcase", "measure", "val")
    )
    df_no_sherlock = df.query("condition == 'view' or condition == 'recall'")
    return pd.concat([df_no_sherlock, sherlock_combined], axis=0).reset_index(drop=True)


@curry
def calc_fd(mat_file, dataset="fnl"):
    """
    Compute Framewise Displacement as in Power et al 2012 and save to file. Uses the default radius of 50mm.
    
    Args:
        mat_file (path object): .mat file outputted from FSL
        dataset (str): 'fnl' or 'sherlock'
        
    Returns:
        file_path
        
    """

    assert isinstance(mat_file, Path), "realigned_series must be a Path object"
    out_file = mat_file.parent / "fd_power_2012.csv"

    fd = FramewiseDisplacement(
        in_file=mat_file, out_file=out_file, parameter_source="FSL", save_plot=False
    )
    if dataset == "fnl":
        fd.inputs.series_tr = 2.0
    elif dataset == "sherlock":
        fd.inputs.series_tr = 1.5

    _ = fd.run()
    return out_file


@curry
def append_fd_to_group_file_with_headcase_id(
    file_path, group_file, additional_no_case=[]
):
    """Load FD csv file, append to a groupfile, and delete the original"""

    df = pd.read_csv(file_path)
    condition = file_path.parent.stem
    subject_id = file_path.parent.parent.stem
    data_id = file_path.parent.parent.parent.stem

    if (
        data_id == "fnl_f17"
        or data_id == "sherlock"
        or subject_id in additional_no_case
    ):
        headcase = "no_case"
    else:
        headcase = "has_case"
    df = df.assign(
        subject_id=subject_id,
        data_id=data_id,
        condition=condition,
        tr=list(range(df.shape[0])),
        headcase=headcase,
    )

    if group_file.exists():
        df.to_csv(group_file, mode="a", header=False, index=False)
    else:
        df.to_csv(group_file, index=False)

    file_path.unlink()


@ed.grps_have_same_nunique_val("subject_id", "measure")
@ed.reset_index()
def summarize_fd_by_subject(df):

    return (
        df
        >> p.group_by("subject_id", "condition", "data_id", "headcase")
        >> p.summarize(
            fd_mean="mean(FramewiseDisplacement)",
            fd_median="median(FramewiseDisplacement)",
            fd_mean_filter="filter_mean(FramewiseDisplacement)",
            fd_median_filter="filter_median(FramewiseDisplacement)",
            perc_spikes="perc_high_motion(FramewiseDisplacement)",
        )
        >> p.do(
            lambda df: df.melt(
                id_vars=["subject_id", "data_id", "condition", "headcase"],
                value_vars=[
                    "fd_mean",
                    "fd_median",
                    "fd_mean_filter",
                    "fd_median_filter",
                    "perc_spikes",
                ],
                var_name="measure",
                value_name="val",
            )
        )
        >> p.arrange("subject_id")
        >> p.call(".reset_index", drop=True)
    )


def calc_motion_deltas(mat_file, diff=True, normalize=False):
    """
    This function reproduces 99% of calculation in nipype.algorithms.confounds.FramewiseDisplacement
    but leaves out the summation step. It can also z-score.
    Output columns are: x, y, z, pitch, roll, yaw
    """

    from nipype.utils.misc import normalize_mc_params
    from scipy.stats import zscore

    mat = np.loadtxt(mat_file)
    mpars = np.apply_along_axis(
        func1d=normalize_mc_params, axis=1, arr=mat, source="FSL"
    )
    if diff:
        mpars = mpars[:-1, :6] - mpars[1:, :6]
        mpars[:, 3:6] *= 50
        mpars = np.abs(mpars)
    else:
        mpars[:, 3:6] *= 50
    if normalize:
        mpars = zscore(mpars)
        
    return mpars, mat_file


@curry
def append_motion_deltas_to_group_file_with_headcase_id(
    mat_and_filepath, group_file, additional_no_case=[]
):
    """
    Take a motion delta numpy array and original filepath and append it to a group file
    """

    # Expand first argument which a tuple of (mat, file_path)
    mat, file_path = mat_and_filepath

    condition = file_path.parent.stem
    subject_id = file_path.parent.parent.stem
    data_id = file_path.parent.parent.parent.stem
    df = pd.DataFrame(mat, columns=["x", "y", "z", "pitch", "roll", "yaw"])
    if (
        data_id == "fnl_f17"
        or data_id == "sherlock"
        or subject_id in additional_no_case
    ):
        headcase = "no_case"
    else:
        headcase = "has_case"
    df = df.assign(
        subject_id=subject_id,
        data_id=data_id,
        condition=condition,
        tr=list(range(df.shape[0])),
        headcase=headcase,
    )

    if group_file.exists():
        df.to_csv(group_file, mode="a", header=False, index=False)
    else:
        df.to_csv(group_file, index=False)


@ed.grps_have_same_nunique_val("subject_id", "measure")
@ed.reset_index()
def summarize_mpars_by_subject(df):
    return (
        df
        >> p.group_by("subject_id", "condition", "data_id", "headcase")
        >> p.summarize(
            x_mean="mean(x)",
            x_median="median(x)",
            x_std="std(x)",
            y_mean="mean(y)",
            y_median="median(y)",
            y_std="std(y)",
            z_mean="mean(z)",
            z_median="median(z)",
            z_std="std(z)",
            pitch_mean="mean(pitch)",
            pitch_median="median(pitch)",
            pitch_std="std(pitch)",
            roll_mean="mean(roll)",
            roll_median="median(roll)",
            roll_std="std(roll)",
            yaw_mean="mean(yaw)",
            yaw_median="median(yaw)",
            yaw_std="std(yaw)",
        )
        >> p.call(
            ".melt",
            id_vars=["subject_id", "data_id", "condition", "headcase"],
            value_vars=[
                "x_mean",
                "y_mean",
                "z_mean",
                "x_median",
                "y_median",
                "z_median",
                "x_std",
                "y_std",
                "z_std",
                "pitch_mean",
                "roll_mean",
                "yaw_mean",
                "pitch_median",
                "roll_median",
                "yaw_median",
                "pitch_std",
                "roll_std",
                "yaw_std",
            ],
            var_name="measure",
            value_name="val",
        )
        >> p.arrange("subject_id")
        >> p.call(".reset_index", drop=True)
    )
