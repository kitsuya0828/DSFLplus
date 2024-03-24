import argparse
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots


def visualize_client_distribution(src_path: str, output_path: str):
    assert os.path.isfile(src_path), f"{src_path} is not a file."

    if output_path is None:
        output_path = os.path.splitext(src_path)[0] + "_partition.png"

    data = pd.read_csv(src_path, index_col=0, header=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={"hspace": 0.05})

    left = [0] * len(data.index)
    for class_id in data.columns:
        counts = data[class_id]

        ax1.barh(data.index, counts, left=left, label=class_id)
        ax2.barh(data.index, counts, left=left, label=class_id)

        left = [sum(x) for x in zip(left, counts)]

    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html

    ax1.set_ylim(94.5, 99.5)
    ax2.set_ylim(-0.5, 5.5)

    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=12,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    plt.legend(title="Class", loc="center left", bbox_to_anchor=(1, 1), borderaxespad=1)
    plt.xlabel("Count")
    fig.text(0.04, 0.5, "Client ID", va="center", rotation="vertical")

    plt.savefig(output_path, bbox_inches="tight")
    plt.close("all")


def visualize_client_distribution_plotly(src_path: str, output_path: str):
    assert os.path.isfile(src_path), f"{src_path} is not a file."

    if output_path is None:
        output_path = os.path.splitext(src_path)[0] + "_partition_plotly.svg"

    data = pd.read_csv(src_path, index_col=0, header=0)

    client_ids = []
    class_labels = []
    counts = []

    for client_id in data.index:
        for class_id in data.columns:
            client_ids.append(client_id)
            class_labels.append(class_id)
            counts.append(data.loc[client_id, class_id])

    sc = go.Scatter(
        x=client_ids,
        y=class_labels,
        mode="markers",
        marker=dict(
            size=counts,
            sizemode="area",
            sizeref=2.0 * max(counts) / (40.0**2),
            sizemin=4,
            color="#1f77b4",
        ),
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.01,
        shared_yaxes=True,
    )

    fig.append_trace(sc, row=1, col=1)
    sc.showlegend = False
    fig.append_trace(sc, row=1, col=2)

    fig.update_xaxes(range=[-0.5, 9.5], row=1, col=1, dtick=1)
    fig.update_xaxes(range=[89.5, 99.5], row=1, col=2, dtick=1)
    fig.update_yaxes(dtick=20)

    fig.update_layout(
        xaxis_title="",
        yaxis_title="Class ID",
        showlegend=False,
    )
    # put text in the middle of two subplots
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.2,
        font=dict(size=16),
        showarrow=False,
        text="Client ID",
    )

    fig.write_image(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--plotly", action="store_true")

    args = parser.parse_args()

    if args.plotly:
        visualize_client_distribution_plotly(
            src_path=args.csv_path, output_path=args.output_path
        )
    else:
        visualize_client_distribution(
            src_path=args.csv_path, output_path=args.output_path
        )
