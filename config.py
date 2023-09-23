import os

from omegaconf import OmegaConf

from args import args

# Base config
base_cfg = OmegaConf.load('configs/base.yaml')

# Main Config
main_cfg = OmegaConf.load(args.config_file)

cfg = OmegaConf.merge(base_cfg, main_cfg)

import torch  # isort:skip # noqa

cfg.checkpoint = f"checkpoints/{cfg.checkpoint}"
dirs = os.listdir(cfg.checkpoint) if os.path.exists(cfg.checkpoint) else []
dirs.sort()
for i, dir in enumerate(dirs):
    assert (dir.startswith('round'))
next_index = len(dirs) + 1

if args.resume:
    cfg.resume = f"{cfg.checkpoint}/round{args.round}"
else:
    cfg.resume = False

if not args.test:
    cfg.checkpoint = f"{cfg.checkpoint}/round{next_index}"
    assert (not os.path.exists(cfg.checkpoint))
    cfg.test = False
else:
    cfg.checkpoint = f"{cfg.checkpoint}/round{args.round}"
    cfg.log_file = "test.txt"
    cfg.test = True

if not os.path.exists(cfg.checkpoint):
    os.makedirs(cfg.checkpoint)
