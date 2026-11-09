"""Small JAX MLP for wavefront reconstruction: model, loss and Adam training.

The network maps standardised lantern signals to standardised aberration
coefficients: dense layers, LayerNorm on the first two hidden layers, Mish
activations and dropout after the second hidden layer (training only).
"""
from __future__ import annotations

import time
from typing import Dict, List, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random


def init_mlp_params(layer_sizes: Sequence[int], key: jax.Array) -> Dict[str, List[jax.Array]]:
    """Glorot-normal weights, zero biases."""
    keys = random.split(key, len(layer_sizes) - 1)
    weights, biases = [], []
    for i in range(len(layer_sizes) - 1):
        in_dim, out_dim = layer_sizes[i], layer_sizes[i + 1]
        stddev = jnp.sqrt(2.0 / (in_dim + out_dim))
        weights.append(random.normal(keys[i], (in_dim, out_dim)) * stddev)
        biases.append(jnp.zeros((out_dim,)))
    return {"weights": weights, "biases": biases}


@jax.jit
def mish(x: jnp.ndarray) -> jnp.ndarray:
    return x * jnp.tanh(jax.nn.softplus(x))


@jax.jit
def layer_norm(x: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(variance + eps)


def mlp_forward(params: Dict, X: jnp.ndarray, key: jax.Array = None,
                deterministic: bool = True, dropout_rate: float = 0.025) -> jnp.ndarray:
    """Forward pass; dropout after the second hidden layer only when
    ``deterministic`` is False and a ``key`` is given."""
    weights, biases = params["weights"], params["biases"]
    num_layers = len(weights)
    dropout_key = key

    for i in range(num_layers - 1):
        X = jnp.dot(X, weights[i]) + biases[i]
        if i < 2:
            X = layer_norm(X)
        X = mish(X)
        if i == 1 and not deterministic and dropout_key is not None:
            dropout_key, subkey = random.split(dropout_key)
            mask = random.bernoulli(subkey, 1.0 - dropout_rate, X.shape)
            X = (X * mask) / (1.0 - dropout_rate)

    return jnp.dot(X, weights[-1]) + biases[-1]


def mlp_forward_np(params: Dict, X: np.ndarray) -> np.ndarray:
    """Inference-only forward pass in plain numpy (no jax dispatch), e.g. to
    time the per-sample latency of a real-time reconstructor on the CPU.
    Same result as ``mlp_forward(params, X)``."""
    weights = [np.asarray(w) for w in params["weights"]]
    biases = [np.asarray(b) for b in params["biases"]]
    X = np.asarray(X, dtype=weights[0].dtype)
    for i in range(len(weights) - 1):
        X = X @ weights[i] + biases[i]
        if i < 2:
            X = (X - X.mean(axis=-1, keepdims=True)) / np.sqrt(X.var(axis=-1, keepdims=True) + 1e-5)
        X = X * np.tanh(np.logaddexp(0.0, X))
    return X @ weights[-1] + biases[-1]


def n_params(layer_sizes: Sequence[int]) -> int:
    """Number of weights and biases of an MLP with these layer sizes."""
    return int(sum((a + 1) * b for a, b in zip(layer_sizes[:-1], layer_sizes[1:])))


@jax.jit
def huber_loss(y_true: jnp.ndarray, y_pred: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
    error = y_true - y_pred
    abs_error = jnp.abs(error)
    quadratic = jnp.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return jnp.mean(0.5 * jnp.square(quadratic) + delta * linear)


def train_mlp(layer_sizes: Sequence[int], X_train, Y_train, X_val, Y_val, *,
              epochs: int = 500, batch_size: int = 128, learning_rate: float = 3e-4,
              lr_schedule: str = "constant", init_seed: int = 0, train_seed: int = 1,
              log_every: int = 100, init_params: Dict | None = None,
              history: Dict | None = None, eval_train: bool = False):
    """Train with Adam on minibatches reshuffled every epoch (the last partial
    batch is dropped). Each epoch is one jit-compiled ``lax.scan`` over its
    minibatches, so the data stays on the device and Python only loops over
    epochs.

    lr_schedule : "constant", or "cosine" (decays from learning_rate to 1% of
        it over the whole run).

    init_params, history : continue an existing run. The epochs are appended to
        ``history`` and the epoch with the lowest validation loss of the whole
        run is kept (for the epochs already in ``history`` only the returned
        parameters are available, so the best epoch of the new part counts from
        the weights handed in).

    eval_train : also record the training loss without dropout, on the whole
        training set, in ``history["train_eval"]``. The plain ``train`` entry is
        the mean minibatch loss *with* dropout switched on, so it sits above the
        validation loss; ``train_eval`` is the like-for-like curve.

    Returns ``(params, history)``: the parameters of the epoch with the lowest
    validation loss, and ``{"train": [...], "val": [...], "best_epoch": int,
    "seconds": float}`` (epochs counted from 1).
    """
    X_train, Y_train = jnp.asarray(X_train), jnp.asarray(Y_train)
    X_val, Y_val = jnp.asarray(X_val), jnp.asarray(Y_val)
    if init_params is None:
        params = init_mlp_params(list(layer_sizes), random.PRNGKey(init_seed))
    else:
        params = init_params
    n_steps = X_train.shape[0] // batch_size
    if lr_schedule == "cosine":
        lr = optax.cosine_decay_schedule(learning_rate, decay_steps=epochs * n_steps, alpha=0.01)
    elif lr_schedule == "constant":
        lr = learning_rate
    else:
        raise ValueError(f"lr_schedule must be 'constant' or 'cosine', not {lr_schedule!r}")
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    def step(carry, idx):
        params, opt_state, key = carry
        key, sub = random.split(key)
        loss, grads = jax.value_and_grad(
            lambda p: huber_loss(Y_train[idx], mlp_forward(p, X_train[idx], key=sub,
                                                           deterministic=False)))(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state, key), loss

    @jax.jit
    def run_epoch(params, opt_state, key):
        key, perm_key = random.split(key)
        batches = random.permutation(perm_key, X_train.shape[0])[:n_steps * batch_size]
        (params, opt_state, key), losses = jax.lax.scan(
            step, (params, opt_state, key), batches.reshape(n_steps, batch_size))
        val = huber_loss(Y_val, mlp_forward(params, X_val))
        return params, opt_state, key, losses.mean(), val

    key = random.PRNGKey(train_seed)
    train_hist = list(history["train"]) if history else []
    val_hist = list(history["val"]) if history else []
    eval_hist = list(history.get("train_eval", [])) if history else []
    best, best_val = params, np.inf
    best_epoch = int(history.get("best_epoch", 0)) if history else 0
    if val_hist:                        # continuing: the weights handed in are the best so far
        best_val = float(min(val_hist[-1], val_hist[-1]))
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        params, opt_state, key, tr, va = run_epoch(params, opt_state, key)
        tr, va = float(tr), float(va)
        train_hist.append(tr)
        val_hist.append(va)
        if eval_train:
            eval_hist.append(float(huber_loss(Y_train, mlp_forward(params, X_train))))
        if va < best_val:
            best, best_val, best_epoch = params, va, len(val_hist)
        if log_every and (epoch % log_every == 0 or epoch == 1):
            print(f"  epoch {epoch:5d}  train {tr:.3e}  val {va:.3e}  "
                  f"({time.time() - t0:.0f} s)")
    history = {"train": train_hist, "val": val_hist, "best_epoch": best_epoch,
               "seconds": time.time() - t0}
    if eval_hist:
        history["train_eval"] = eval_hist
    return best, history


def save_mlp(path: str, params: Dict, layer_sizes: Sequence[int], **extra) -> None:
    """Save the weights (as plain numpy) and ``extra`` metadata to an ``.npz``."""
    arrays = {f"w{i}": np.asarray(w) for i, w in enumerate(params["weights"])}
    arrays.update({f"b{i}": np.asarray(b) for i, b in enumerate(params["biases"])})
    np.savez(path, layer_sizes=np.asarray(layer_sizes), **arrays,
             **{f"extra_{k}": np.asarray(v) for k, v in extra.items()})


def load_mlp(path: str):
    """Inverse of save_mlp: returns ``(params, layer_sizes, extra)``."""
    with np.load(path) as f:
        layer_sizes = [int(n) for n in f["layer_sizes"]]
        n = len(layer_sizes) - 1
        params = {"weights": [jnp.asarray(f[f"w{i}"]) for i in range(n)],
                  "biases": [jnp.asarray(f[f"b{i}"]) for i in range(n)]}
        extra = {k[6:]: f[k] for k in f.files if k.startswith("extra_")}
    return params, layer_sizes, extra
