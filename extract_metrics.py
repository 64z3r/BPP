#! /usr/bin/env python

import click
import pandas as pd


@click.command()
@click.argument(
    "dataset",
    type=click.Choice(
        ["train", "val", "test"],
        case_sensitive=False,
    ),
)
@click.argument(
    "input_path",
    type=click.Path(exists=True),
)
@click.argument(
    "output_path",
    type=click.Path(),
)
def extract_metrics(dataset, input_path, output_path):
    """Extract metrics."""

    columns_dataset = [
        f"{dataset}/f1",
        f"{dataset}/ppv",
        f"{dataset}/tpr",
        f"{dataset}/iou",
    ]
    columns = ["epoch", "step", *columns_dataset]
    df = pd.read_csv(input_path)
    df_extract = (
        df[columns]
        .groupby(["epoch", "step"], group_keys=True)
        .apply(lambda group: group.ffill())
        .drop_duplicates(subset=["epoch", "step"])
        .dropna(subset=columns_dataset, how="all")
        .reset_index(drop=True)
    )
    df_extract.to_csv(output_path, index=False)

    print(f"Transformation completed. Output saved to {output_path}")


if __name__ == "__main__":
    extract_metrics()
