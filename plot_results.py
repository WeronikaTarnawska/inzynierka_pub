# ==========================================================================================
#                         This file is part of the Bachelor Thesis project
#                                  University of Wrocław
#                        Author: Weronika Tarnawska (Index No. 331171)
#                                        June 2025
# ==========================================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def barplot(
    data,
    x,
    y,
    hue=None,
    order=None,
    title=None,
    filename=None,
    time_plot=False,
):
    """Draws the plot."""
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=data, x=x, y=y, hue=hue, order=order, errorbar=None)
    plt.title(title)
    label = (
        "Time [s]"
        if time_plot
        else ("SNR improvement [dB]" if y == "snr_db" else "SNR improvement")
    )
    plt.ylabel(label)
    plt.xlabel(x)
    plt.xticks(rotation=45)

    # move legend outside to the right (won't cover the bars)
    # bbox_to_anchor=(1.02, 1) - anchor point for the legend (top right corner of the plot)
    # loc='upper left' - which part of the legend is anchored at that point
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def create_param_label(row):
    """Generates parameter description label for plot legend."""
    alg = row["algorithm"]
    if alg in ["SS-Magnitude", "SS-Power"]:
        return f"frame {str(row['frame'])[:-5]}k, alpha {row['alpha']}"
    elif alg in ["Wiener-Inst", "Wiener-Avg"]:
        return f"frame {str(row['frame'])[:-5]}k"
    elif alg == "LMS":
        return f"filter {str(row['filter'])[:-2]}, step {row['step']}"
    else:
        return ""


if __name__ == "__main__":

    # set style settings
    sns.set(style="whitegrid")
    plt.rcParams.update({"figure.dpi": 200})

    # load data
    df = pd.read_csv("results.csv")

    # convert to numeric values
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce", downcast="integer")
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["filter"] = pd.to_numeric(df["filter"], errors="coerce", downcast="integer")

    # create column with parameter description (param_label)
    df["param_label"] = df.apply(create_param_label, axis=1)

    # prepare folders for plots (if they don't exist)
    os.makedirs("img/snr", exist_ok=True)
    os.makedirs("img/time", exist_ok=True)

    # for each case draw three plots: SNR and Time
    cases = df["case"].unique()
    for case in cases:
        case_df = df[df["case"] == case]
        if case_df.empty:
            continue

        # SNR improvement
        barplot(
            case_df,
            x="algorithm",
            y="snr_db",
            hue="param_label",
            title=f"{case}: SNR",
            filename=f"img/snr/{case}_snr.png",
        )

        # Time
        barplot(
            case_df,
            x="algorithm",
            y="time_sec",
            hue="param_label",
            time_plot=True,
            title=f"{case}: Time",
            filename=f"img/time/{case}_time.png",
        )
