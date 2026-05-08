from risk_bridge import build_scenario1_run_config, run_simulation


def main() -> None:
  cfg = build_scenario1_run_config(
    nsim=2,
    n_target=1000,
    n_source=500,
    n_reference=1000,
    sample_size=100,
    output_root="data",
  )
  run_dir = run_simulation(cfg)
  print(f"Wrote Risk Bridge outputs to {run_dir}")


if __name__ == "__main__":
  main()
