#!/usr/bin/env python3
"""
orbax.py

Minimal, practical Orbax examples in Python:

- save_pytree: save an arbitrary pytree (dict of arrays) with orbax.checkpoint
- restore_pytree: restore a checkpoint
- rotate_checkpoints: simple retention policy (keep last N)
- example_flax_usage: how you'd save/restore a Flax TrainState (optional)

Usage:
  python orbax_example.py --demo          # run a demo save -> restore -> rotate
  python orbax_example.py --save PATH     # save sample state to PATH
  python orbax_example.py --restore PATH  # restore from PATH and print keys
"""

import os
import argparse
import time
import shutil
from typing import Any, Dict, List

import numpy as np
import jax
import jax.numpy as jnp

# Orbax
import orbax.checkpoint as ocp


def make_sample_state(step: int = 0) -> Dict[str, Any]:
    """Create a small sample pytree to checkpoint."""
    return {
        "step": np.array(step, dtype=np.int64),
        "params": {
            "dense/kernel": np.arange(12, dtype=np.float32).reshape(3, 4),
            "dense/bias": np.arange(3, dtype=np.float32),
        },
        "opt_state": {
            "adam_m": np.zeros((3, 4), dtype=np.float32),
            "adam_v": np.ones((3, 4), dtype=np.float32),
        },
    }


def save_pytree(ckpt_path: str, pytree: Any, overwrite: bool = False) -> None:
    """
    Save a pytree with Orbax.

    Args:
      ckpt_path: directory path (orbax will create files inside, e.g. ckpt_path/)
      pytree: any JAX/NumPy pytree (dicts, lists, arrays)
      overwrite: if True and ckpt_path exists, remove it first
    """
    handler = ocp.PyTreeCheckpointHandler()
    checkpointer = ocp.Checkpointer(handler)

    if os.path.exists(ckpt_path):
        if overwrite:
            shutil.rmtree(ckpt_path)
        else:
            raise FileExistsError(f"Checkpoint path already exists: {ckpt_path}")

    os.makedirs(ckpt_path, exist_ok=True)
    # Orbax commonly uses a directory per checkpoint. We'll write into ckpt_path.
    print(f"[save_pytree] Saving checkpoint to: {ckpt_path}")
    checkpointer.save(ckpt_path, pytree)
    print("[save_pytree] Save complete.")


def restore_pytree(ckpt_path: str) -> Any:
    """
    Restore a pytree with Orbax.

    Returns the restored pytree object.
    """
    handler = ocp.PyTreeCheckpointHandler()
    checkpointer = ocp.Checkpointer(handler)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[restore_pytree] Restoring checkpoint from: {ckpt_path}")
    restored = checkpointer.restore(ckpt_path)
    print("[restore_pytree] Restore complete.")
    return restored


def list_checkpoints(parent_dir: str) -> List[str]:
    """Return a sorted list of checkpoint subdirs in parent_dir (most recent last)."""
    if not os.path.isdir(parent_dir):
        return []
    items = [os.path.join(parent_dir, p) for p in os.listdir(parent_dir)]
    ckpts = [p for p in items if os.path.isdir(p)]
    # Sort by mtime to approximate creation order
    ckpts.sort(key=lambda p: os.path.getmtime(p))
    return ckpts


def rotate_checkpoints(parent_dir: str, keep_last: int = 3) -> None:
    """
    Keep only the most recent `keep_last` checkpoint directories under parent_dir.
    Deletes older checkpoint directories.

    Note: This is a small helper for local filesystems. For cloud storage, adapt accordingly.
    """
    ckpts = list_checkpoints(parent_dir)
    if len(ckpts) <= keep_last:
        print(f"[rotate_checkpoints] Nothing to delete (have {len(ckpts)}, keep {keep_last}).")
        return

    to_delete = ckpts[:-keep_last]
    for path in to_delete:
        print(f"[rotate_checkpoints] Removing old checkpoint: {path}")
        shutil.rmtree(path)
    print("[rotate_checkpoints] Rotation complete.")


# Optional: Example of saving/restoring a Flax TrainState (common in JAX+Flax workflows)
def example_flax_usage(ckpt_path: str, overwrite: bool = True) -> None:
    """
    Demonstrates saving a Flax TrainState. This function will only run if Flax is installed.
    Flax's TrainState is a dataclass and is a pytree — Orbax can checkpoint it directly.
    """
    try:
        from flax.training.train_state import TrainState
        import optax
        import flax.linen as nn
    except Exception as e:
        print("[example_flax_usage] Flax not available; skip the TrainState demo.", e)
        return

    # minimal flax model + train state
    class DummyDense(nn.Module):
        features: int = 4

        @nn.compact
        def __call__(self, x):
            return nn.Dense(self.features)(x)

    # init params
    rng = jax.random.PRNGKey(0)
    model = DummyDense(features=4)
    dummy_x = jnp.ones((1, 3), dtype=jnp.float32)
    params = model.init(rng, dummy_x)

    tx = optax.adam(1e-3)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Save
    save_pytree(ckpt_path, state, overwrite=overwrite)

    # Restore
    restored = restore_pytree(ckpt_path)
    # Note: restored will be a TrainState-like pytree. You can inspect restored.params etc.
    print("[example_flax_usage] Restored TrainState keys:", list(restored.__dict__.keys()))


def demo_flow(root_dir: str = "/tmp/orbax_demo", keep_last: int = 3) -> None:
    """
    A simple demo:
    - create three checkpoints with incremental step numbers
    - list them
    - restore the most recent
    - rotate to keep last `keep_last`
    """
    os.makedirs(root_dir, exist_ok=True)
    ckpt_dirs = []
    for i in range(1, 4):
        ckpt_dir = os.path.join(root_dir, f"ckpt_{i:03d}")
        state = make_sample_state(step=i * 100)
        # create unique path for this demo run (orbax writes files into the dir)
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        save_pytree(ckpt_dir, state)
        # tiny sleep to ensure distinct mtimes
        time.sleep(0.1)
        ckpt_dirs.append(ckpt_dir)

    print("\n[demo_flow] All checkpoints:")
    for p in list_checkpoints(root_dir):
        print("  ", p)

    # restore latest
    latest = list_checkpoints(root_dir)[-1]
    restored = restore_pytree(latest)
    print("[demo_flow] Restored step:", int(restored["step"]))

    # rotate
    print("\n[demo_flow] Rotating to keep last", keep_last)
    rotate_checkpoints(root_dir, keep_last=keep_last)
    print("[demo_flow] After rotation, checkpoints:")
    for p in list_checkpoints(root_dir):
        print("  ", p)


def main():
    parser = argparse.ArgumentParser(description="Orbax Python examples")
    parser.add_argument("--demo", action="store_true", help="Run the demo flow")
    parser.add_argument("--save", type=str, help="Save a sample state to PATH (dir will be created)")
    parser.add_argument("--restore", type=str, help="Restore checkpoint from PATH and print keys")
    parser.add_argument("--flax-demo", type=str, help="Run the Flax TrainState demo and save to PATH")
    args = parser.parse_args()

    if args.demo:
        demo_flow()
        return

    if args.save:
        sample = make_sample_state(step=42)
        save_pytree(args.save, sample, overwrite=True)
        return

    if args.restore:
        restored = restore_pytree(args.restore)
        print("Top-level keys:", list(restored.keys()))
        if "step" in restored:
            print("Restored step:", int(restored["step"]))
        return

    if args.flax_demo:
        example_flax_usage(args.flax_demo)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
