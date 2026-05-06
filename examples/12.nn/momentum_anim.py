"""
momentum_anim.py
================
Animate the difference between plain Gradient Descent and Gradient Descent
with Momentum on an elongated ("ravine") loss surface.

Loss used:
    L(w1, w2) = 0.05 * w1**2 + 2.0 * w2**2

The w2 direction is much steeper than the w1 direction, so plain GD with a
moderate learning rate oscillates back-and-forth across the ravine while
making slow progress along the gentle w1 direction.  Momentum smooths these
oscillations because the velocity averages out alternating gradients.

Update rules (matching the lecture slide):
    Vanilla GD : W = W - alpha * grad(L)
    Momentum   : v = beta * v + (1 - beta) * grad(L)     # smoothed gradient
                 W = W - alpha * v

Run:
    python momentum_anim.py            # saves momentum_anim.gif (default)
    python momentum_anim.py --show     # try to open an interactive window
    python momentum_anim.py --mp4      # save momentum_anim.mp4 (needs ffmpeg)
"""

import sys
import numpy as np
import matplotlib

# ------------------------------------------------------------------
# Pick a backend BEFORE importing pyplot.
# Default: "Agg" (headless, never tries to talk to a display server).
# Only attempt an interactive backend if the user passes --show.
# ------------------------------------------------------------------
_SHOW = "--show" in sys.argv
_MP4  = "--mp4"  in sys.argv
if not _SHOW:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt              # noqa: E402  (after matplotlib.use)
from matplotlib.animation import FuncAnimation


# ----------------------------------------------------------------------
# 1. Loss surface and its gradient
# ----------------------------------------------------------------------
def loss(w1, w2):
    # Elongated bowl: gentle along w1, steep along w2  -> a "ravine".
    return 0.05 * w1**2 + 2.0 * w2**2


def grad(w1, w2):
    # Partial derivatives: dL/dw1, dL/dw2.
    return np.array([0.10 * w1, 4.0 * w2])


# ----------------------------------------------------------------------
# 2. Optimizers
# ----------------------------------------------------------------------
def run_gd(start, lr, n_steps):
    """Plain gradient descent: W <- W - alpha * grad."""
    w = np.array(start, dtype=float)
    path = [w.copy()]
    for _ in range(n_steps):
        w = w - lr * grad(*w)
        path.append(w.copy())
    return np.array(path)


def run_momentum(start, lr, beta, n_steps):
    """GD + Momentum (EMA-of-gradients form, as on the lecture slide).

        v_t = beta * v_{t-1} + (1 - beta) * grad   # blend old velocity & new grad
        W   = W - alpha * v_t                      # step using smoothed grad
    """
    w = np.array(start, dtype=float)
    v = np.zeros_like(w)                       # velocity, starts from rest
    path = [w.copy()]
    for _ in range(n_steps):
        v = beta * v + (1.0 - beta) * grad(*w)
        w = w - lr * v
        path.append(w.copy())
    return np.array(path)


# ----------------------------------------------------------------------
# 3. Hyperparameters chosen so the contrast is easy to see
# ----------------------------------------------------------------------
START   = (-4.5, 1.0)   # same starting point for both methods
LR      = 0.40          # large enough to make plain GD oscillate in w2
BETA    = 0.90          # momentum coefficient (matches the lecture example)
N_STEPS = 60

path_gd  = run_gd(START, LR, N_STEPS)
path_mom = run_momentum(START, LR, BETA, N_STEPS)

loss_gd  = np.array([loss(*p) for p in path_gd])
loss_mom = np.array([loss(*p) for p in path_mom])


# ----------------------------------------------------------------------
# 4. Figure with two panels:  contour + trajectories  |  loss vs step
# ----------------------------------------------------------------------
xs = np.linspace(-5.0, 5.0, 300)
ys = np.linspace(-1.5, 1.5, 300)
X, Y = np.meshgrid(xs, ys)
Z = loss(X, Y)

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(13, 5),
    gridspec_kw={"width_ratios": [2, 1]},
)

# --- Left panel: contours of the loss surface --------------------------
ax1.contour(X, Y, Z, levels=20, cmap="viridis", alpha=0.6)
ax1.plot(0, 0, "k*", ms=14, label="minimum")
ax1.set_xlim(xs.min(), xs.max())
ax1.set_ylim(ys.min(), ys.max())
ax1.set_xlabel(r"$w_1$  (gentle direction)")
ax1.set_ylabel(r"$w_2$  (steep direction)")
ax1.set_title("Trajectory on the loss surface")

line_gd,  = ax1.plot([], [], "o-", color="crimson",   lw=2, ms=4,
                     label=f"Vanilla GD  (α={LR})")
line_mom, = ax1.plot([], [], "o-", color="royalblue", lw=2, ms=4,
                     label=f"Momentum  (α={LR}, β={BETA})")
ax1.legend(loc="lower right")

# --- Right panel: loss curves -----------------------------------------
ax2.set_xlim(0, N_STEPS)
ax2.set_ylim(0, max(loss_gd.max(), loss_mom.max()) * 1.05)
ax2.set_xlabel("step")
ax2.set_ylabel("loss")
ax2.set_title("Loss over time")
loss_line_gd,  = ax2.plot([], [], color="crimson",   lw=2, label="Vanilla GD")
loss_line_mom, = ax2.plot([], [], color="royalblue", lw=2, label="Momentum")
ax2.legend(loc="upper right")
ax2.grid(alpha=0.3)

# Live numeric readout on the contour panel.
text_box = ax1.text(
    0.02, 0.97, "", transform=ax1.transAxes,
    va="top", ha="left", family="monospace",
    bbox=dict(boxstyle="round", fc="white", alpha=0.85),
)


# ----------------------------------------------------------------------
# 5. Animation update function
# ----------------------------------------------------------------------
def update(i):
    # Reveal the trajectories one step at a time.
    line_gd.set_data(path_gd[:i + 1, 0],  path_gd[:i + 1, 1])
    line_mom.set_data(path_mom[:i + 1, 0], path_mom[:i + 1, 1])

    steps = np.arange(i + 1)
    loss_line_gd.set_data(steps,  loss_gd[:i + 1])
    loss_line_mom.set_data(steps, loss_mom[:i + 1])

    text_box.set_text(
        f"step      : {i:3d}\n"
        f"GD   loss : {loss_gd[i]:6.3f}\n"
        f"Mom. loss : {loss_mom[i]:6.3f}"
    )
    return line_gd, line_mom, loss_line_gd, loss_line_mom, text_box


ani = FuncAnimation(
    fig, update, frames=N_STEPS + 1,
    interval=120, blit=True, repeat=False,
)

fig.suptitle(
    "Gradient Descent vs Gradient Descent + Momentum",
    fontsize=14, fontweight="bold",
)
plt.tight_layout()


# ----------------------------------------------------------------------
# 6. Render: file by default, interactive window only if --show
# ----------------------------------------------------------------------
if _SHOW:
    # User explicitly asked for a window. If the chosen GUI backend has
    # no display available (common on WSL/SSH), matplotlib will raise --
    # we let the error surface so the user sees the real cause.
    plt.show()
elif _MP4:
    out_path = "momentum_anim.mp4"
    print(f"[info] saving {out_path} (requires ffmpeg on PATH)...")
    ani.save(out_path, writer="ffmpeg", fps=10)
    print(f"[info] saved {out_path}")
else:
    out_path = "momentum_anim.gif"
    print(f"[info] backend = {matplotlib.get_backend()}; "
          f"saving animation to {out_path} ...")
    ani.save(out_path, writer="pillow", fps=10)   # requires: pip install pillow
    print(f"[info] saved {out_path}  (open it in VS Code / a browser)")
