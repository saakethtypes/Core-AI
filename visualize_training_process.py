from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam, SGD
from tqdm import tqdm

#Function to learn
def rosenbrock(xy):
    x, y = xy
    return (x+(2*y)-7)**2+((2*x)+y-5)**2

#Training
def run_optimization(xy_init, optimizer_class, n_iter, **optimizer_kwargs):
    xy_t = torch.tensor(xy_init, requires_grad=True)
    optimizer = optimizer_class([xy_t], **optimizer_kwargs)

    path = np.empty((n_iter + 1, 2))
    path[0, :] = xy_init

    for i in tqdm(range(1, n_iter + 1)):
        optimizer.zero_grad()
        loss = rosenbrock(xy_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(xy_t, 1.0)
        optimizer.step()

        path[i, :] = xy_t.detach().numpy()

    return path

#Animation
def create_animation(paths,
                     colors,
                     names,
                     figsize=(22, 22),
                     x_lim=(-6, 6),
                     y_lim=(-2, 6),
                     n_seconds=12):

    if not (len(paths) == len(colors) == len(names)):
        raise ValueError

    path_length = max(len(path) for path in paths)

    n_points = 300
    x = np.linspace(*x_lim, n_points)
    y = np.linspace(*y_lim, n_points)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock([X, Y])

    minimum = (1.0, 1.0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.contour(X, Y, Z, 90, cmap="jet")

    scatters = [ax.scatter(None,
                           None,
                           label=label,
                           c=c) for c, label in zip(colors, names)]

    ax.legend(prop={"size": 25})
    ax.plot(*minimum, "rD")

    def animate(i):
        for path, scatter in zip(paths, scatters):
            scatter.set_offsets(path[:i, :])

        ax.set_title(str(i))

    ms_per_frame = 1000 * n_seconds / path_length

    anim = FuncAnimation(fig, animate, frames=path_length, interval=ms_per_frame)

    return anim

if __name__ == "__main__":
    xy_init = (.3, .8)
    n_iter = 2500

    path_adam = run_optimization(xy_init, Adam, n_iter)
    path_sgd = run_optimization(xy_init, SGD, n_iter, lr=1e-3)

    freq = 10

    paths = [path_adam[::freq], path_sgd[::freq]]
    colors = ["green", "blue"]
    names = ["Adam", "SGD"]

    anim = create_animation(paths,
                            colors,
                            names,
                            figsize=(48, 28),
                            x_lim=(-.1, 1.1),
                            y_lim=(-.1, 1.1),
                            n_seconds=12)

    anim.save("result.gif")
