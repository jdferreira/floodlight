from datetime import date
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import seaborn as sns


# @st.cache
def load_data():
    data = pd.read_csv("user-migration.csv", parse_dates=["day"])

    data["day"] = data["day"].dt.date

    return data.set_index("day")


def plot_history(data):
    fig, ax = plt.subplots(figsize=(8, 3))

    data = data.assign(
        fraction=lambda df: df["in_both"] / (df["in_both"] + df["in_app_b"]),
    )

    sns.lineplot(data=data, x="day", y="fraction", ax=ax)
    ax.set_ylabel("")
    ax.set_xlabel("")

    plt.xticks(rotation=90)
    st.pyplot(fig)


def plot_venn_diagram(day_stats):
    fig, ax = plt.subplots()

    venn2(
        {
            "10": day_stats["in_app_a"],
            "01": day_stats["in_app_b"],
            "11": day_stats["in_both"],
        },
        set_labels=("App A", "App B"),
        ax=ax,
    )

    st.pyplot(fig)


if __name__ == "__main__":
    st.title("User migration analysis")

    data = load_data()

    st.markdown(
        """Company XZBT is launching an new application that will supercede a
        previous one they had build before. They want to keep track of the
        migration of users from "App A" to "App B"."""
    )

    st.markdown(
        """The first graph shows the trend of the fraction of users in App B
        that are also in App A."""
    )
    plot_history(data)

    st.header("Venn diagrams")
    st.markdown(
        """We can also see a slice two slices of time: select two dates on the
        slider and see a venn diagram of the users distributed among the apps
        they use for both dates."""
    )

    dates_selected = st.slider(
        "Select two dates",
        data.index.min(),
        data.index.max(),
        (data.index.min(), data.index.max()),
    )

    col1, col2 = st.columns(2)
    with col1:
        plot_venn_diagram(data.loc[dates_selected[0]])
    with col2:
        plot_venn_diagram(data.loc[dates_selected[1]])
