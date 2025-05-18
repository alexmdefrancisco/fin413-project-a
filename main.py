import argparse
import yaml

from src.part1 import prepare_data
from src.part2 import compute_covariances
from src.part3 import build_risk_portfolios
from src.part4 import build_hrpe, build_tsm_hrpe, compute_performance

def main(step: str, cfg_path: str = "config.yaml"):
    # load config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    lr = None

    if step in ("all", "part1"):
        lr = prepare_data(cfg)

    if step in ("all", "part2"):
        if lr is None:
            lr = prepare_data(cfg)
        compute_covariances(cfg, lr)          # produces and saves cov_PP/Tr

    if step in ("all", "part3"):
        if lr is None:
            lr = prepare_data(cfg)
        build_risk_portfolios(cfg, lr)        # MV, ERC, ENB, HRP

    if step in ("all", "part4"):
        if lr is None:
            lr = prepare_data(cfg)
        build_hrpe(cfg, lr)                   # HRPe (SW distance)
        build_tsm_hrpe(cfg, lr)               # TSM-HRPe
        compute_performance(cfg, lr)          # performance of 1/N, HRPe, TSM-HRPe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run project steps")
    parser.add_argument(
        "--step",
        choices=["all", "part1", "part2", "part3", "part4"],
        default="all",
        help="Which stage to run",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()
    main(args.step, args.config)