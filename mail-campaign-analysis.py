from datetime import date, timedelta
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import dates as mdates
from matplotlib import patches as mpatches
from millify import millify
from scipy import stats


@st.cache
def load_data(campaign: str):
    campaign = {
        "Campaign 1": "a",
        "Campaign 2": "b",
        "Dummy campaign": "dummy",
    }[campaign]

    num_journeys = pd.read_csv(f"data/campaign-{campaign}-num-journeys.csv")
    spendings = pd.read_csv(f"data/campaign-{campaign}-spendings.csv")

    for df in [num_journeys, spendings]:
        df.columns = [
            date.fromisoformat(col) if col != "group" else "group"
            for col in df.columns
        ]

    mail_date = {
        "a": date(2022, 5, 15),
        "b": date(2022, 6, 1),
        "dummy": date(2022, 1, 1),
    }[campaign]

    n_control = {
        "a": 500,
        "b": 1000,
        "dummy": 500,
    }[campaign]

    n_treated = {
        "a": 500,
        "b": 2500,
        "dummy": 500,
    }[campaign]

    return {
        "mail_date": mail_date,
        "n_control": n_control,
        "n_treated": n_treated,
        "num_journeys": num_journeys,
        "spendings": spendings,
    }


def get_longevity_bounds(pvalues, mail_date, pvalue_threshold, non_significant_stride_threshold):
    """
    Given a map from days to pvalues (associated with testing whether the
    average number of journeys or spendings is greater on the treated group than
    in the control group on that day), and a p-value threshold, find the "best"
    interval where the campaign was sucessful.
    """

    first_significant = last_significant = None

    # The first time we see a second consecutive `False`, that's the end of the effect
    for day, pvalue in pvalues.items():
        if day >= mail_date and pvalue <= pvalue_threshold:
            if first_significant is None:
                first_significant = day

            prev_significant = day
            non_significant_stride = 0
        elif first_significant is not None:
            non_significant_stride += 1

            if non_significant_stride == non_significant_stride_threshold:
                last_significant = prev_significant
                break

    if last_significant is None and first_significant is not None:
        last_significant = day

    return (first_significant, last_significant)


def run_analysis(data, mail_date, n_control, n_treated, p=None, start=None, end=None):
    """
    Run the ananlysis on the data, either selecting the interval where the
    campaign had its effect automatically (using `p` as the p-value threshold
    for that), or using `start` and `end` as the limits of that tinterval
    """

    grouped = (
        data.groupby("group")
        .mean()
        .T.melt(ignore_index=False, value_name="num_journeys")
        .rename_axis(index="day")
        .reset_index()
    )

    pvalues = (
        data.groupby("group")
        .agg(list)
        .T.apply(
            lambda row: stats.ttest_ind(
                row["treated"], row["control"], alternative="greater"
            ).pvalue,
            axis=1,
        )
    )

    if p is not None:
        start, end = get_longevity_bounds(pvalues, mail_date, p, 2)

    try:
        longevity = end - start
    except:
        longevity = timedelta(days=0)

    if longevity > timedelta(days=0):
        treatment_total_outcome = (
            data[data["group"] == "treated"].drop(columns=["group"]).T.loc[start:end].sum().sum()
        )
        baseline = (
            data[data["group"] == "control"].drop(columns=["group"]).T.loc[start:end].sum().sum()
            / n_control
            * n_treated
        )
    else:
        treatment_total_outcome = 0
        baseline = 0

    return {
        "grouped": grouped,
        "pvalues": pvalues,
        "start": start,
        "end": end,
        "longevity": longevity,
        "baseline": baseline,
        "treatment_total_outcome": treatment_total_outcome,
    }


st.title("Mail campaign analysis")

st.markdown(
    """
    This dashboard analyses the effect of email campaigns, both form the point
    of view of the **number of journeys** made by the users in the two different
    groups and the **total spendings** of these users.
    """
)

campaign = st.sidebar.selectbox("Choose a campaign", ["Campaign 1", "Campaign 2", "Dummy campaign"])

data = load_data(campaign)

st.markdown(
    f"""
    **Sent date:** {data["mail_date"]}

    This dashboard tries to determine the interval on which the campaign had an
    effect, using the a level of statistical significance to find this interval.
    You can overide this on the sidebar, in which case you will be asked to
    select a start and end date manually.
    """
)

with st.sidebar:
    auto_longevity = st.checkbox("Automatic longevity analysis?", True)
    if auto_longevity:
        pvalue_threshold_text = st.text_input("Significance level", "0.01")
        try:
            p: Optional[float] = float(pvalue_threshold_text)
        except ValueError:
            p = None
        if p is None or p <= 0 or p > 1:
            st.error(f"Please use a valid significance level")
        else:
            num_journeys_analysis = run_analysis(
                data["num_journeys"],
                data["mail_date"],
                data["n_control"],
                data["n_treated"],
                p=p
            )
            spendings_analysis = run_analysis(
                data["spendings"],
                data["mail_date"],
                data["n_control"],
                data["n_treated"],
                p=p
            )

    else:
        start_input = st.date_input("Start", data["mail_date"])
        end_input = st.date_input("End", data["mail_date"])

        num_journeys_analysis = run_analysis(
            data["num_journeys"],
            data["mail_date"],
            data["n_control"],
            data["n_treated"],
            start=start_input,
            end=end_input,
        )
        spendings_analysis = run_analysis(
            data["spendings"],
            data["mail_date"],
            data["n_control"],
            data["n_treated"],
            start=start_input,
            end=end_input,
        )

st.header("Summary of conclusions")
num_journeys_metric, spendings_metric = st.columns(2)
num_journeys_metric.metric(
    "# journeys",
    millify(num_journeys_analysis["treatment_total_outcome"]),
    millify(num_journeys_analysis["treatment_total_outcome"] - num_journeys_analysis["baseline"]),
)
spendings_metric.metric(
    "spendings",
    millify(spendings_analysis["treatment_total_outcome"]),
    millify(spendings_analysis["treatment_total_outcome"] - spendings_analysis["baseline"]),
)


def plot(analysis, *, ylabel):
    fig, ax = plt.subplots(figsize=(10, 4))
    twinx = ax.twinx()

    sns.set_style("ticks")

    sns.lineplot(
        data=analysis["grouped"],
        x="day",
        y="num_journeys",
        hue="group",
        palette={"control": "lightgray", "treated": "green"},
        ax=ax,
    )

    sns.lineplot(
        data=analysis["pvalues"],
        ax=twinx,
        color="darkblue",
        linewidth=1,
    )

    ax.set_ylim(0, None)
    ax.set_xticks([data["mail_date"] + timedelta(days=i) for i in (-14, -7, 0, 7, 14, 21, 28)], rotation=90)
    ax.set_xlabel("Day")
    ax.set_ylabel(ylabel)
    ax.legend(title="Group")

    ax.axvline(data["mail_date"], color="green", linewidth=3.0)

    twinx.set_yscale("log")
    twinx.set_ylim(None, 10**10)

    if auto_longevity:
        twinx.axhline(p, alpha=0.3, color="darkblue")
    twinx.set_yticks([10**i for i in range(-4, 1)])

    if auto_longevity and analysis["start"] is not None:
        rect = mpatches.Rectangle(
            (analysis["start"], 0),
            analysis["longevity"],
            p,
            fill="darkblue",
            alpha=0.3,
        )
        twinx.add_patch(rect)

        xy = []
        points = analysis["grouped"].set_index(["group", "day"])

        for i in range(0, analysis["longevity"].days + 1):
            day = analysis["start"] + timedelta(days=i)
            xy.append((mdates.date2num(day), points.loc["control", day].values[0]))

        for i in range(analysis["longevity"].days, -1, -1):
            day = analysis["start"] + timedelta(days=i)
            xy.append((mdates.date2num(day), points.loc["treated", day].values[0]))

        ax.add_patch(mpatches.Polygon(xy, edgecolor="green", hatch="...", fill=False))

    sns.despine(offset={"left": 10})
    st.pyplot(fig)

    # if longevity > timedelta(days=0):
    #     measured_effect = (
    #         grouped.pivot(index="day", columns="group")
    #         .droplevel(0, axis=1)
    #         .loc[first_significant:last_significant]
    #         .assign(difference=lambda df: df["treated"] - df["control"])["difference"]
    #         .sum()
    #         * N_TREATED
    #     )

    #     baseline = (
    #         grouped.pivot(index="day", columns="group")
    #         .droplevel(0, axis=1)
    #         .loc[first_significant:last_significant]
    #         .assign(difference=lambda df: df["control"])["control"]
    #         .sum()
    #         * N_TREATED
    #     )

    #     st.metric(f"Effect on {variable_selection}", baseline, round(measured_effect, 1))

    #     st.markdown(
    #         f"""
    #         Based on the analysis on the "{variable_selection}" for the campaign
    #         `{campaign}`, this campaign had a measurable effect from {first_significant}
    #         to {last_significant}, and lead to an increase in the {variable_selection}
    #         of {measured_effect:.1f} throughout that period.
    #     """
    #     )
    # else:
    #     st.markdown("No significant effect observed for this campaign")


st.header("Plots")

st.subheader("Number of journeys")
plot(num_journeys_analysis, ylabel="Average # journeys per user")

st.subheader("Spendings")
plot(spendings_analysis, ylabel="Average spendings per user")
