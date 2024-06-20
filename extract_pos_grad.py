#! /usr/bin/env python

import click
import pandas as pd


@click.command()
@click.argument(
    "input_path",
    type=click.Path(exists=True),
)
@click.argument(
    "output_path",
    type=click.Path(),
)
def extract_pos_grad(input_path, output_path):
    """Extract coordinate head gradient norms."""

    df = pd.read_csv(input_path)
    columns_pos_grad = list(
        filter(lambda name: name.startswith("grad_") and "pos_net" in name, df.columns)
    )
    columns_pos_grad.sort()
    columns = ["epoch", "step", *columns_pos_grad]
    df_extract = (
        df[columns]
        .groupby(["epoch", "step"], group_keys=True)
        .apply(lambda group: group.ffill())
        .drop_duplicates(subset=["epoch", "step"])
        .dropna(subset=columns_pos_grad, how="all")
        .reset_index(drop=True)
    )
    df_extract.to_csv(output_path, index=False)

    print(f"Transformation completed. Output saved to {output_path}")


if __name__ == "__main__":
    extract_pos_grad()
