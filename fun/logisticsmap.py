import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sympy import diff, lambdify, latex, pi, roots, sin, symbols

x, r = symbols("x, r", real = True)

plt.style.use("dark_background")

# Any function with roots at 0 and 1, and a maximum in-between
# with a value of 0.25; exclude `r`, it'll be added later!
fun = x * (1 - x)

grw_res = 8192 # points on r-axis
all_its = 4096 # iterations in total
plt_its = 625  # iterations to plot
xmin = 1
xmax = 4

A = np.zeros((grw_res, plt_its))
A[:, -1] = 0.5

dfn = diff(fun, x)
roots = []

try:
    rts = np.array(roots(dfn, x))
except:
    rts = np.array([0.5,])

logistic = lambdify(x, fun)
eps = 1e-16
rts = rts[rts > -eps]
rts = rts[rts < 1.0+eps]
rts = rts[logistic(rts) > -eps]
peak = np.amax(logistic(rts))

logistic = lambdify([r, x], 0.25 / peak * r * fun)

eqn = latex(fun)
plt.figure(figsize=(36, 9))
plt.title("${0}$".format(eqn), fontsize=16)

for i in tqdm(np.arange(1, grw_res)):
    r = xmin + ((xmax - xmin) * i) / grw_res
    A[i, 0] = logistic(r, A[i, -1])

    for n in range(all_its - plt_its):
        A[i, 0] = logistic(r, A[i, 0])

    for j in np.arange(1, plt_its):
        A[i, j] = logistic(r, A[i, j - 1])

    x = r * np.ones(plt_its)
    plt.scatter(x, A[i, :], s=0.04, c="white", marker=",", alpha=0.02)

plt.xlim([xmin, xmax])
plt.ylim([0, 1])
plt.savefig("logistic.svg", dpi=900, bbox_inches="tight")