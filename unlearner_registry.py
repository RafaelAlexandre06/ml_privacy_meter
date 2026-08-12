"""Registry and guardrails for the U-RMIA unlearner interface.

Kept separate from ``unlearners.py`` so that an unlearner living in its own
module can self-register without an import cycle. The cycle is real: importing
``unlearners`` triggers its bottom-of-file imports of the external unlearner
modules, so if one of those did ``from unlearners import register`` at module
level and happened to be imported *first*, it would find ``unlearners`` only
partially initialized. This module imports nothing from the project, so both
directions are safe.

Adding an unlearner
-------------------
Write a function with the interface documented in ``unlearners.py`` and decorate
it, declaring the ``unlearn.params`` keys it reads::

    @register("my_method", OPTIMIZER_PARAMS | {"epochs", "alpha"})
    def my_method(model, forget_loader, retain_loader,
                  unlearn_configs, train_configs, logger):
        ...

If it lives in its own module, add an import for that module at the bottom of
``unlearners.py`` so the decorator runs. That is the whole registration.

The decorator declares parameter *names* only; defaults stay in the function
body next to the code (and the comment) that explains them, so the declaration
and the default cannot drift apart. What the names buy is
``validate_unlearn_params``: a misspelled key in a config is otherwise
indistinguishable from an absent one, silently falls back to the default, and
produces a run that looks exactly like a correct one.

Cache guards
------------
The other silent-failure mode is reuse. A cached ``.pkl`` or signal array from a
*different* unlearner has the same shape as the right one, so nothing downstream
notices. ``check_cached_unlearn`` and ``signal_cache_matches`` compare the stored
``unlearn_fingerprint`` instead. Both treat *missing* provenance as a match, so
runs produced before the fingerprint existed keep resuming; only a fingerprint
that is present and disagrees is an error.
"""

import difflib
import json
import logging
import os

# Keys consumed by ``unlearners._optimizer_config``. Any unlearner that
# fine-tunes reads these, so it unions them into its own declaration.
OPTIMIZER_PARAMS = frozenset(
    {"optimizer", "learning_rate", "weight_decay", "momentum"}
)

# Recognised keys of the ``unlearn`` config block itself (as opposed to the
# per-algorithm ``params`` sub-block). ``seed`` is filled in from
# ``run.random_seed`` by the callers when absent.
UNLEARN_BLOCK_KEYS = frozenset(
    {"algorithm", "forget_size", "params", "seed", "allow_stale_cache"}
)

UNLEARNERS = {}


def register(name: str, params=frozenset()):
    """Decorator registering an unlearner under ``name``.

    Args:
        name (str): The value ``unlearn.algorithm`` must take to select it.
        params (Iterable[str]): The ``unlearn.params`` keys this unlearner
            reads. Anything else in a config is rejected by
            ``validate_unlearn_params``.

    Raises:
        ValueError: If ``name`` is already registered, which would otherwise
            silently shadow one implementation with another.

    Returns:
        Callable: The decorator.
    """

    def decorator(fn):
        if name in UNLEARNERS and UNLEARNERS[name] is not fn:
            raise ValueError(
                f"Unlearner '{name}' is already registered to "
                f"{UNLEARNERS[name].__module__}.{UNLEARNERS[name].__name__}."
            )
        fn.unlearn_name = name
        fn.declared_params = frozenset(params)
        UNLEARNERS[name] = fn
        return fn

    return decorator


def get_unlearner(name: str):
    """Look up an unlearning algorithm by name.

    Args:
        name (str): The ``unlearn.algorithm`` config value.

    Raises:
        NotImplementedError: If the algorithm is not registered.

    Returns:
        Callable: The unlearner function following the module's interface.
    """
    if name not in UNLEARNERS:
        raise NotImplementedError(
            f"Unlearning algorithm '{name}' is not implemented. "
            f"Supported: {sorted(UNLEARNERS)}"
        )
    return UNLEARNERS[name]


def _suggest(key: str, candidates) -> str:
    """Return a ' (did you mean ...?)' hint for a mistyped key, or ''."""
    close = difflib.get_close_matches(key, sorted(candidates), n=1, cutoff=0.6)
    return f" (did you mean '{close[0]}'?)" if close else ""


def validate_unlearn_params(configs: dict) -> None:
    """Reject ``unlearn`` config keys the selected unlearner does not read.

    Unknown keys are the one config error this pipeline cannot survive quietly:
    every unlearner reads its settings with ``params.get(key, default)``, so a
    typo produces the default and a run that is wrong but entirely plausible.

    Args:
        configs (dict): Full config dictionary.

    Raises:
        ValueError: If the ``unlearn`` block or its ``params`` sub-block
            contains a key the selected unlearner does not declare.
    """
    block = configs["unlearn"]
    unknown_block = sorted(set(block) - UNLEARN_BLOCK_KEYS)
    if unknown_block:
        raise ValueError(
            "Unknown key(s) in the 'unlearn' config block: "
            + ", ".join(f"'{k}'{_suggest(k, UNLEARN_BLOCK_KEYS)}" for k in unknown_block)
            + f". Recognised: {sorted(UNLEARN_BLOCK_KEYS)}."
        )

    unlearner = get_unlearner(block["algorithm"])
    declared = getattr(unlearner, "declared_params", None)
    if declared is None:
        # Registered without the decorator; nothing to validate against.
        return

    params = block.get("params") or {}
    unknown = sorted(set(params) - set(declared))
    if unknown:
        raise ValueError(
            f"Unknown unlearn.params key(s) for algorithm '{block['algorithm']}': "
            + ", ".join(f"'{k}'{_suggest(k, declared)}" for k in unknown)
            + f". Accepted: {sorted(declared)}."
        )


def unlearn_fingerprint(configs: dict) -> dict:
    """The settings that determine what an unlearned model *is*.

    Two runs agreeing on this fingerprint produce interchangeable unlearned
    models; two that disagree do not, no matter how identical the array shapes
    look. This is the single definition used by every cache guard below.

    Args:
        configs (dict): Full config dictionary.

    Returns:
        dict: ``{"algorithm": str, "params": dict}``.
    """
    block = configs["unlearn"]
    return {
        "algorithm": block["algorithm"],
        "params": dict(block.get("params") or {}),
    }


def _stored_fingerprint(stored_meta: dict):
    """Read a fingerprint out of saved model metadata, or None if absent.

    ``None`` means the artifact predates fingerprinting, not that it disagrees.
    """
    if "unlearn_algorithm" not in stored_meta and "unlearn_params" not in stored_meta:
        return None
    return {
        "algorithm": stored_meta.get("unlearn_algorithm"),
        "params": dict(stored_meta.get("unlearn_params") or {}),
    }


def check_cached_unlearn(
    stored_meta: dict,
    role: str,
    configs: dict,
    pkl_path: str,
    logger: logging.Logger,
) -> None:
    """Refuse to reuse a cached model that was unlearned with other settings.

    The resume path exists so a long run can be restarted; it is not a way to
    swap algorithms. Because a stale ``.pkl`` has the right architecture and the
    signals cache keys only on array shape, reusing one produces a complete,
    well-formed audit *of the previous unlearner*. That has already cost a run,
    which is why this raises rather than warns.

    Set ``unlearn.allow_stale_cache: true`` to go back to warning, for the case
    where the mismatch is known to be cosmetic.

    Args:
        stored_meta (dict): The saved metadata entry for ``role``.
        role (str): Model role, for the message.
        configs (dict): Full config dictionary.
        pkl_path (str): Path of the cached checkpoint, named in the message.
        logger (logging.Logger): Logger object for the current run.

    Raises:
        ValueError: On a fingerprint mismatch, unless the escape hatch is set.
    """
    stored = _stored_fingerprint(stored_meta)
    current = unlearn_fingerprint(configs)
    if stored is None or stored == current:
        return

    message = (
        f"Cached model '{role}' was unlearned with {stored} but the config asks "
        f"for {current}. Reusing it would audit the previous unlearner: the "
        f"checkpoint has the right architecture and the signal cache keys only "
        f"on array shape, so nothing downstream would notice. Either set "
        f"run.log_dir to a new directory (recommended, keeps both runs), or "
        f"delete {pkl_path} together with the cached arrays in "
        f"{configs['run']['log_dir']}/signals. Set unlearn.allow_stale_cache: "
        f"true to reuse it anyway."
    )
    if configs["unlearn"].get("allow_stale_cache", False):
        logger.warning("allow_stale_cache is set. %s", message)
        return
    raise ValueError(message)


def _sidecar_path(signal_path: str) -> str:
    return f"{signal_path}.meta.json"


def signal_cache_matches(
    signal_path: str, configs: dict, logger: logging.Logger
) -> bool:
    """Whether a cached signal array was computed under the current unlearn config.

    The shape check the signal caches already do cannot see this: swapping the
    unlearner changes every value in the ``unlearned`` column and none of the
    dimensions.

    A **missing** sidecar counts as a match. Signal arrays written before this
    check existed have no provenance to compare, and treating them as stale
    would silently invalidate every run already on disk.

    Args:
        signal_path (str): Path of the cached ``.npy``.
        configs (dict): Full config dictionary.
        logger (logging.Logger): Logger object for the current run.

    Returns:
        bool: True if the cache may be reused.
    """
    sidecar = _sidecar_path(signal_path)
    if not os.path.exists(sidecar):
        return True
    try:
        with open(sidecar, "r") as f:
            stored = json.load(f)
    except (OSError, ValueError):
        logger.warning("Could not read %s; treating the cache as stale.", sidecar)
        return False

    current = unlearn_fingerprint(configs)
    if stored == current:
        return True
    logger.warning(
        "Cached signals %s were computed with unlearn settings %s but the config "
        "has %s; recomputing.",
        signal_path,
        stored,
        current,
    )
    return False


def write_signal_fingerprint(signal_path: str, configs: dict) -> None:
    """Record the unlearn settings a signal array was computed under.

    Call this next to every ``np.save`` of a signal cache, and also after
    accepting an un-fingerprinted legacy array, so the next run has something to
    compare against.
    """
    with open(_sidecar_path(signal_path), "w") as f:
        json.dump(unlearn_fingerprint(configs), f, indent=4)
