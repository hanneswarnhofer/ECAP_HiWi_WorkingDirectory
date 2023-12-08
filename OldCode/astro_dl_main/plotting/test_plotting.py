from matplotlib import pyplot as plt
from plotting.style import mplrc
import numpy as np
from tools.utils import config


CONFIG = config()

mplrc(True)
x = np.random.randn(10)
y = np.random.randn(10)


fig, ax = plt.subplots(1)
ax.scatter(x, y, label="$data$")

x = np.random.randn(10)
y = np.random.randn(10)
ax.scatter(x, y, color="red", label="data red")

ax.set_title(r"$\alpha$")
ax.set_ylabel("$y / meters$")
ax.set_xlabel("$x / meters$")
ax.legend()

fig.savefig(CONFIG.log_dir + "/test_scatter.pgf")

fig.savefig(CONFIG.log_dir + "/test_scatter.png", dpi=180)
fig.savefig(CONFIG.log_dir + "/test_scatter.pdf")
