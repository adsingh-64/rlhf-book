"""Sweep wrapper for preference RM training.

Reads hyperparams from wandb.config (set by the sweep) and calls train_preference_rm.

Usage:
    # 1. Create the sweep
    wandb sweep reward_models/sweep.yaml

    # 2. Launch an agent
    wandb agent <sweep-id>
"""

import wandb

from reward_models.train_preference_rm import train_preference_rm

MAX_MICRO_BATCH = 8


def main():
    wandb.init()
    config = wandb.config

    effective_bs = config.effective_batch_size
    micro_batch = min(effective_bs, MAX_MICRO_BATCH)
    grad_accum = effective_bs // micro_batch

    train_preference_rm(
        samples=config.samples,
        batch_size=micro_batch,
        grad_accum_steps=grad_accum,
        epochs=config.epochs,
        lr=config.lr,
        use_wandb=True,
    )


if __name__ == "__main__":
    main()
