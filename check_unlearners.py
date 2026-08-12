"""Conformance harness for the unlearner interface.

Run this against a new unlearner before spending GPU on it::

    python check_unlearners.py

CPU-only, a few seconds, no dataset download and no checkpoints. It exercises
every algorithm in ``UNLEARNERS`` on a tiny synthetic fixture and checks the
parts of the contract the function signature cannot express:

* the return value is an ``nn.Module``, on CPU, with the same ``state_dict``
  keys it was handed -- everything downstream reloads it with
  ``model.load_state_dict``, which fails or silently mismatches otherwise;
* **the input model is not mutated**. ``urmia_utils.py`` passes the live
  ``models["original"]``, so an unlearner that forgets ``copy.deepcopy``
  corrupts that role in place. Nothing in the pipeline can detect this: the
  audit still completes, and the ``original`` row -- the positive control the
  whole run is judged against -- is quietly wrong;
* it is deterministic under a fixed seed, which is what makes a resumed or
  repeated run comparable;
* the declared ``params`` and the config validator agree, in both directions.

It also *reports*, without failing, whether an unlearner's randomness is
controlled by ``unlearn.seed`` or merely inherited from ambient global RNG
state. Both are reproducible run-to-run, so neither is wrong; but an
ambient-dependent one changes its output when anything upstream changes how much
randomness it consumed -- an extra reference model, a reordered loader -- so two
"identical" configs can disagree. ``gaussian_noise``, ``random_label`` and
``wig`` take a local ``torch.Generator``; the gradient-ascent methods inherit the
DataLoader shuffling, which is ambient by construction.

The fixture is deliberately hostile to shortcuts: two classes, unbalanced
forget/retain sizes, and a model carrying Conv2d, Linear and BatchNorm so that
``wig``'s per-layer weight selection has something to score and so any unlearner
touching running statistics shows up in the mutation check.
"""

import copy
import inspect
import logging
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from unlearner_registry import UNLEARNERS, UNLEARN_BLOCK_KEYS, validate_unlearn_params
import unlearners  # noqa: F401  (imported for its @register side effects)

SEED = 12345
N_FORGET, N_RETAIN = 24, 72
NUM_CLASSES = 2

# Enough epochs to actually move weights, few enough to stay instant.
PARAMS = {
    "gaussian_noise": {"noise_std": 0.01},
    "neggrad": {"epochs": 1, "learning_rate": 0.001},
    "random_label": {"epochs": 1, "learning_rate": 0.001},
    "neggrad_plus": {"epochs": 1, "learning_rate": 0.001},
    "wig": {"epochs": 1, "learning_rate": 0.001, "init_ratio": 0.3,
            "grad_batch_size": 8},
}

logging.basicConfig(level=logging.WARNING, format="      %(levelname)s %(message)s")
logger = logging.getLogger("check_unlearners")


class TinyNet(nn.Module):
    """Conv + BatchNorm + Linear: the layer types the unlearners discriminate on."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4 * 8 * 8, NUM_CLASSES)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x)))
        return self.fc(x.flatten(1))


def make_fixture():
    """A fresh model and the two loaders, all from the same fixed seed."""
    torch.manual_seed(SEED)
    model = TinyNet()
    forget = TensorDataset(
        torch.randn(N_FORGET, 3, 8, 8), torch.randint(0, NUM_CLASSES, (N_FORGET,))
    )
    retain = TensorDataset(
        torch.randn(N_RETAIN, 3, 8, 8), torch.randint(0, NUM_CLASSES, (N_RETAIN,))
    )
    # shuffle=False so a failure means the unlearner is non-deterministic, not
    # that the batch order moved.
    return (
        model,
        DataLoader(forget, batch_size=8, shuffle=False),
        DataLoader(retain, batch_size=8, shuffle=False),
    )


def snapshot(model):
    """A detached CPU copy of every tensor, buffers included.

    ``state_dict`` is used rather than ``parameters`` on purpose: BatchNorm
    running statistics are buffers, and an unlearner that runs a forward pass on
    a shared model in ``train()`` mode mutates them without touching a weight.
    """
    return {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}


def identical(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def run_one(name, fn):
    """Check one unlearner.

    Returns:
        tuple[list[str], list[str]]: contract violations, then informational
        notes. An empty violations list means it conforms.
    """
    failures = []
    notes = []
    configs = {
        "run": {"log_dir": "unused", "random_seed": SEED},
        "train": {"device": "cpu", "optimizer": "SGD", "weight_decay": 0.0,
                  "momentum": 0.0, "batch_size": 8, "model_name": "tiny"},
        "unlearn": {"algorithm": name, "forget_size": N_FORGET,
                    "params": dict(PARAMS.get(name, {}))},
    }
    unlearn_configs = dict(configs["unlearn"])
    unlearn_configs["seed"] = SEED

    model, forget_loader, retain_loader = make_fixture()
    before = snapshot(model)
    torch.manual_seed(SEED)
    out = fn(model, forget_loader, retain_loader, unlearn_configs,
             configs["train"], logger)

    if not isinstance(out, nn.Module):
        return [f"returned {type(out).__name__}, not an nn.Module"], notes

    devices = {p.device.type for p in out.state_dict().values() if p.is_floating_point()}
    if devices - {"cpu"}:
        failures.append(f"returned a model on {sorted(devices)}, expected CPU only")

    after_out = snapshot(out)
    if set(after_out) != set(before):
        missing = sorted(set(before) - set(after_out))
        extra = sorted(set(after_out) - set(before))
        failures.append(f"state_dict keys changed (missing {missing}, extra {extra})")

    # The one that silently corrupts a run.
    if not identical(before, snapshot(model)):
        changed = [k for k in before
                   if not torch.equal(before[k], snapshot(model)[k])]
        failures.append(
            f"MUTATED the input model in place ({len(changed)} tensors, e.g. "
            f"{changed[:3]}) -- add copy.deepcopy(model) before touching it"
        )

    # Determinism: same seed, same fixture, same weights out.
    model2, f2, r2 = make_fixture()
    torch.manual_seed(SEED)
    out2 = fn(model2, f2, r2, unlearn_configs, configs["train"], logger)
    if not identical(after_out, snapshot(out2)):
        failures.append("not reproducible: two runs at the same seed differ")

    # Informational: is the randomness seed-controlled, or ambient? Same
    # unlearn.seed, different global RNG state. A difference is not a defect --
    # see the module docstring -- but it decides whether two runs of one config
    # can be compared when anything upstream of the unlearner changed.
    model3, f3, r3 = make_fixture()
    torch.manual_seed(SEED + 1)
    out3 = fn(model3, f3, r3, unlearn_configs, configs["train"], logger)
    if not identical(after_out, snapshot(out3)):
        notes.append(
            "output depends on ambient global RNG state, not only on "
            "unlearn.seed (fine, but two runs agree only if everything before "
            "the unlearner drew the same randomness)"
        )

    # Declared params must be the ones the config validator accepts.
    declared = getattr(fn, "declared_params", None)
    if declared is None:
        failures.append("registered without @register, so params are unvalidated")
    else:
        bogus = dict(configs["unlearn"])
        bogus["params"] = dict(bogus["params"])
        bogus["params"]["epocs"] = 1
        try:
            validate_unlearn_params({**configs, "unlearn": bogus})
            failures.append("validator accepted the bogus params key 'epocs'")
        except ValueError:
            pass
        try:
            validate_unlearn_params(configs)
        except ValueError as exc:
            failures.append(f"validator rejected this harness's own params: {exc}")

        # Drift heuristic: a declared name nobody reads is either a typo in the
        # declaration or a parameter that was removed from the body.
        try:
            source = inspect.getsource(fn)
        except OSError:
            source = ""
        if source:
            unread = sorted(k for k in declared if f'"{k}"' not in source
                            and f"'{k}'" not in source)
            # OPTIMIZER_PARAMS are read indirectly, via _optimizer_config.
            unread = [k for k in unread if "_optimizer_config" not in source]
            if unread:
                failures.append(f"declares params never read in its body: {unread}")

    return failures, notes


def main():
    print(f"Checking {len(UNLEARNERS)} registered unlearner(s) against the "
          f"interface contract.\n")

    total = 0
    for name in sorted(UNLEARNERS):
        fn = UNLEARNERS[name]
        where = f"{fn.__module__}.{fn.__name__}"
        if name not in PARAMS:
            print(f"  ? {name:<16} {where}")
            print(f"      no fixture params in check_unlearners.PARAMS; running "
                  f"with defaults")
        failures, notes = run_one(name, fn)
        total += len(failures)
        mark = "OK" if not failures else "FAIL"
        print(f"  {mark:<4} {name:<16} {where}")
        for f in failures:
            print(f"      - {f}")
        for n in notes:
            print(f"      ~ {n}")

    # Block-level keys are validated too, so a mistyped 'seed' cannot go quiet.
    try:
        validate_unlearn_params(
            {"unlearn": {"algorithm": sorted(UNLEARNERS)[0], "seeed": 1}}
        )
        print("\n  FAIL unlearn-block validation accepted the bogus key 'seeed'")
        total += 1
    except ValueError as exc:
        assert "seed" in str(exc), exc
        print(f"\n  OK   unlearn-block keys validated "
              f"({len(UNLEARN_BLOCK_KEYS)} recognised, typos suggested)")

    if total:
        print(f"\n{total} contract violation(s).")
        return 1
    print("\nAll unlearners conform.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
