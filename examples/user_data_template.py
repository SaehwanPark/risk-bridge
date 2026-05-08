import pandas as pd

from risk_bridge import UserDataRunConfig, UserDataSchema, run_user_data


def main() -> None:
  target_df = pd.read_csv("target.csv")
  source_df = pd.read_csv("source.csv")
  reference_df = pd.read_csv("reference.csv")

  config = UserDataRunConfig(
    target_df=target_df,
    source_df=source_df,
    reference_df=reference_df,
    schema=UserDataSchema(
      x_cols=("X1", "X2", "X3", "X4"),
      y_col="caseY",
      z_origin_col="zOrigin",
      z_cat_col="zCat",
    ),
    sample_size=500,
    output_root="data",
    run_label="user_data_example",
  )
  run_dir = run_user_data(config)
  print(f"Wrote Risk Bridge outputs to {run_dir}")


if __name__ == "__main__":
  main()
