# main.py
import argparse, yaml
from src.part1 import prepare_data
from src.part2 import compute_covariances, analyze_covariances
from src.part3 import build_risk_portfolios

def main(step, cfg_path="config.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    lr = None
    if step in ("all", "part1"):
        lr = prepare_data(cfg)

    if step in ("all", "part2"):
        lr = lr or prepare_data(cfg)
        cov_pp, cov_tr = compute_covariances(cfg, lr)

    if step in ("all", "analyze_cov"):
        analyze_covariances(cfg)

    if step in ("all", "part3"):
        lr = lr or prepare_data(cfg)
        build_risk_portfolios(cfg, lr)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--step", choices=["all","part1","part2","analyze_cov","part3"],
                   default="all", help="Which stage to run")
    p.add_argument("--config", default="config.yaml", help="Path to config file")
    args = p.parse_args()
    main(args.step, args.config)