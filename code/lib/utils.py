"""
Functions shared across all n**.py scripts
"""

import numpy as np
import pandas as pd
from rpy2.robjects import r
from pathlib import PurePath, Path
from toolz import curry


@curry
def write_df_to_docx(tup, filename, overwrite=False):
    """
    Take a pandas dataframe and write it into a docx file using the officer and flextable R packages. If the file doesn't exist it'll be created otherwise it'll be appended to. Curry decorator enables this function to be used in a pipe, e.g. toolz.pipe(df, somefunc, write_df_to_docx('myfile.docx'))
    
    Officer docs: https://davidgohel.github.io/officer/articles/offcran/word.html
    Flextable docs: https://davidgohel.github.io/flextable/articles/layout.html
    
    Args:
        tup (tuples): tuple containing df to be converted to table and list of in-line strings 
        filename (str/path): string or path-object ending in '.docx'
        overwrite (bool): whether to first erase the file before writing dataframes
    """

    df, inlines = tup
    assert df.shape[0] == len(
        inlines
    ), f"Number of table rows and inlines strings don't match; nrows: {df.shape[0]} strings: {len(inlines)}"

    r_func = """
    function(df,inlines,filename,pgwidth=6.5) {
        library(magrittr)
        library(officer)
        library(flextable)

        fit_to_page <- function(ft, pgwidth = 6.6){
            ft_out <- ft %>% autofit()
            ft_out <- width(ft_out, width = dim(ft_out)$widths*pgwidth /(flextable_dim(ft_out)$widths))
            return(ft_out)
        }
        ft = flextable(df) %>% bold(~ p.perm < .05, ~ p.perm, bold = TRUE) %>% italic(part="header")
        ft = fit_to_page(ft, pgwidth=pgwidth)
        if (file.exists(filename)) {
            # If the file exists, append to it and place an line above the table
            doc = read_docx(path=filename) %>% body_add_par('', style = 'Normal') %>% body_add_flextable(ft) %>% body_add_par('', style = 'Normal')
        } else {
            # If the file doesn't exist create it and only write a table a
            doc = read_docx() %>% body_add_flextable(ft) %>% body_add_par('', style = 'Normal')
        }
        for (p in inlines) {
            doc = doc %>% body_add_par(p, style='Normal') %>% body_add_par('', style='Normal')
        }
        print(doc, target=filename)
    }
   """

    call_r_func = r(r_func)
    if isinstance(filename, PurePath):
        filename = str(filename.resolve())
    if not Path(filename).exists():
        msg = f"Created new file and successfully inserted Dataframe"
    else:
        if overwrite:
            Path(filename).unlink()
            msg = f"Overwriting existing file. Successfully inserted Dataframe"
        else:
            msg = f"Existing file found. Successfully appended Dataframe"
    call_r_func(df, inlines, filename)
    print(msg)


def make_formatted_df_and_strings_for_writing(df):
    """
    Creates a formatted dataframe in the exact specification it should appear as a table in the manuscript by renaming and resorting columns and string names. Also creates a readable sentence representation of each row of the dataframe that can serve as scaffold for in-text statistics reporting.

    Args:
        df (pd.DataFrame): DataFrame of pairwise comparisons

    Returns:
        formatted_df (pd.DataFrame): DataFrame with renamed and resorted columns
        inlines: list of strings with the same number of rows as formatted_df
    """

    columns = ["Comparison", "Condition", "Metric", "Mean.Difference", "t", "p.perm"]
    name_dict = {
        "fnl_no_case": "FNL no-case",
        "fnl_has_case": "FNL with-case",
        "sherlock_no_case": "Sherlock",
    }
    condition_dict = {"view": "Viewing", "recall": "Talking"}
    metric_dict = {
        "fd_mean": "FD Mean",
        "fd_median": "FD Median",
        "perc_spikes": "Spike Proportion",
        "fd_mean_filter": "FD MeanFiltered",
        "fd_median_filter": "FD MedianFiltered",
    }
    formatted_df = []
    inlines = []
    for i, doc in df.iterrows():
        names = doc.comparison.split("__")
        names = [name_dict[e] for e in names]
        condition = condition_dict[doc.condition]
        metric = metric_dict[doc.measure]
        name1, name2 = names
        comparison = f"{name1} - {name2}"
        mdiff = np.round(doc["mean_diff"], 3)
        cil, ciu = np.round(doc["ci_lower"], 3), np.round(doc["ci_upper"], 3)
        mdiff_formatted = f"{mdiff}\n({cil} {ciu})"
        mean1 = np.round(doc["x_mean"], 3)
        mean2 = np.round(doc["y_mean"], 3)
        sd1 = np.round(doc["x_std"], 3)
        sd2 = np.round(doc["y_std"], 3)
        pval = np.round(doc["perm_p"], 3)
        pstring = f"p = {str(pval)[1:]}" if pval > 0.001 else "p < .001"
        tstat = np.round(doc["t-stat"], 3)
        result_row = dict(
            zip(columns, [comparison, condition, metric, mdiff_formatted, tstat, pval])
        )
        inline = f"{metric}, {name1} (M = {mean1}; SD = {sd1}) and {name2} (M = {mean2}; SD = {sd2}) t = {tstat}, {pstring}"

        formatted_df.append(result_row)
        inlines.append(inline)

    formatted_df = pd.DataFrame(formatted_df)

    return (formatted_df, inlines)


def make_formatted_mpars_df_and_strings_for_writing(df):
    """
    Creates a formatted dataframe in the exact specification it should appear as a table in the manuscript by renaming and resorting columns and string names. Also creates a readable sentence representation of each row of the dataframe that can serve as scaffold for in-text statistics reporting.

    Args:
        df (pd.DataFrame): DataFrame of pairwise comparisons

    Returns:
        formatted_df (pd.DataFrame): DataFrame with renamed and resorted columns
        inlines: list of strings with the same number of rows as formatted_df
    """

    columns = ["Motion Parameter", "Metric", "Mean.Difference", "t", "p.perm"]
    split_rot_metric_dict = {
        "x_mean": ["X", "Mean"],
        "x_median": ["X", "Median"],
        "x_std": ["X", "SD"],
        "y_mean": ["Y", "Mean"],
        "y_median": ["Y", "Median"],
        "y_std": ["Y", "SD"],
        "z_mean": ["Z", "Mean"],
        "z_median": ["Z", "Median"],
        "z_std": ["Z", "SD"],
        "pitch_mean": ["Pitch", "Mean"],
        "pitch_median": ["Pitch", "Median"],
        "pitch_std": ["Pitch", "SD"],
        "roll_mean": ["Roll", "Mean"],
        "roll_median": ["Roll", "Median"],
        "roll_std": ["Roll", "SD"],
        "yaw_mean": ["Yaw", "Mean"],
        "yaw_median": ["Yaw", "Median"],
        "yaw_std": ["Yaw", "SD"],
    }
    reindex = [6, 9, 15, 0, 3, 12, 7, 10, 16, 1, 4, 13, 8, 11, 17, 2, 5, 14]
    df = df.reindex(reindex).reset_index(drop=True)
    formatted_df = []
    inlines = []
    for i, doc in df.iterrows():
        param, metric = split_rot_metric_dict[doc.measure]
        mdiff = np.round(doc["mean_diff"], 3)
        cil, ciu = np.round(doc["ci_lower"], 3), np.round(doc["ci_upper"], 3)
        mdiff_formatted = f"{mdiff}\n({cil} {ciu})"
        mean1 = np.round(doc["x_mean"], 3)
        mean2 = np.round(doc["y_mean"], 3)
        sd1 = np.round(doc["x_std"], 3)
        sd2 = np.round(doc["y_std"], 3)
        pval = np.round(doc["perm_p"], 3)
        pstring = f"p = {str(pval)[1:]}" if pval > 0.001 else "p < .001"
        tstat = np.round(doc["t-stat"], 3)
        result_row = dict(
            zip(columns, [param, metric, mdiff_formatted, tstat, pval])
        )
        inline = f"{param}, {metric} Sherlock (M = {mean1}; SD = {sd1}) and FNL (M = {mean2}; SD = {sd2}) t = {tstat}, {pstring}"

        formatted_df.append(result_row)
        inlines.append(inline)

    formatted_df = pd.DataFrame(formatted_df)

    return (formatted_df, inlines)


def annotate_axis(
    ax,
    xstart,
    xend,
    y,
    texts,
    thickness=1.5,
    color="k",
    fontsize=18,
    offset=0.01,
    xycoords="data",
):
    """
    Draw comparison lines and text/stars on a given matplotlib axis.
    
    Args:
        ax (matplotlib.axes): axes to annotate
        xstart (list): list of starting x-coords
        xend (list): list of ending x-coords
        y (list): list of y-coords that determine comparison bar height
        texts (list): list of texts to add at yannot
        width (int): thickness of all comparison bars
        color (str): color of all comparison bars and text/stars
        fontsize (int): size of text/stars
        offset (float): how much higher than y, text/stars should appear
    
    Returns:
        ax: annotated matplotlib axis
    """

    if not isinstance(xstart, list):
        xstart = [xstart]
    if not isinstance(xend, list):
        xend = [xend]
    if not isinstance(y, list):
        y = [y]
    if not isinstance(texts, list):
        texts = [texts]

    assert (
        len(xstart) == len(xend) == len(y) == len(texts)
    ), "All coordinates and annotations need to have the same number of elements"

    for x1, x2, y, t in zip(xstart, xend, y, texts):
        _ = ax.annotate(
            "",
            xy=(x1, y),
            xycoords=xycoords,
            xytext=(x2, y),
            textcoords="data",
            arrowprops=dict(
                arrowstyle="-",
                ec=color,
                connectionstyle="arc3,rad=0",
                linewidth=thickness,
            ),
        )
        if np.abs(x1) == np.abs(x2):
            midpoint = 0
        else:
            midpoint = np.mean([x1, x2])
        _ = ax.text(
            midpoint,
            y + offset,
            t,
            fontsize=fontsize,
            horizontalalignment="center",
            verticalalignment="center",
        )

    return ax
