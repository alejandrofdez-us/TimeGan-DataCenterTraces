
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import dirichlet
from torch.distributions.multivariate_normal import MultivariateNormal

import torch
from sklearn import metrics

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")



def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    N = len(x)
    M = len(y)
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1 # (a^2) * 1/(a^2 + dxx)
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.25 * dxx / a) # torch.exp(- dxx /2 a)
            YY += torch.exp(-0.25 * dyy / a)
            XY += torch.exp(-0.25 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)
    #return torch.mean(XX) + torch.mean(YY) - 2*torch.mean(XY)


def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

m = 100
# sample size
# x_mean = torch.zeros(2)+1
# print ("x_mean: ", x_mean)
# y_mean = torch.zeros(2)
# print ("y_mean: ", y_mean)
#
# x_cov = 2*torch.eye(2) # IMPORTANT: Covariance matrices must be positive definite
# print ("x_cov: ", x_cov)
# y_cov = 3*torch.eye(2) - 1
# print("y_cov: ",y_cov)
# #
# px = MultivariateNormal(x_mean, x_cov)
# qy = MultivariateNormal(y_mean, y_cov)
# x = px.sample([m]).to(device)
# print ("x: ",x)
# y = qy.sample([m*2]).to(device)
# print("y: ",y)
#



x_matrix = np.loadtxt('data/mu_day3_cut.csv', delimiter=",", skiprows=1)[:1000]
y_matrix = np.loadtxt('experiments/alibabacompletocut10-13-2022-21-22/generated_data/sample_0.csv', delimiter=",")
#x_mean = torch.tensor(np.mean(x_matrix, axis=0))
#print (x_mean)
#y_mean = torch.tensor(np.mean(y_matrix, axis=0))
#x_cov = torch.tensor(np.cov(x_matrix))
#x_cov = 2*torch.eye(4)
#print (x_cov)
#y_cov = torch.tensor(np.cov(y_matrix))
#y_cov = 3*torch.eye(4) - 1

x = torch.tensor(x_matrix).to(device)
y = torch.tensor(y_matrix).to(device)
#result = MMD(x, y, kernel="rbf")

#print(f"MMD result of X and Y is {result.item()}")

result2 = mmd_rbf(x, y)

print ("Result sklearn: ", result2)

# ---- Plotting setup ----
#
# fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8,4), dpi=100)
# #plt.tight_layout()
# delta = 0.025
#
# x1_val = np.linspace(-5, 5, num=m)
# x2_val = np.linspace(-5, 5, num=m)
#
# x1, x2 = np.meshgrid(x1_val, x2_val)
#
# px_grid = torch.zeros(m,m)
# qy_grid = torch.zeros(m,m)
#
#
# for i in range(m):
#     for j in range(m):
#         px_grid[i,j] = multivariate_normal.pdf([x1_val[i],x2_val[j]], x_mean, x_cov)
#         qy_grid[i,j] = multivariate_normal.pdf([x1_val[i],x2_val[j]], y_mean, y_cov)
#
#
# CS1 = ax1.contourf(x1, x2, px_grid,100, cmap=plt.cm.YlGnBu)
# ax1.set_title("Distribution of $X \sim P(X)$")
# ax1.set_ylabel('$x_2$')
# ax1.set_xlabel('$x_1$')
# ax1.set_aspect('equal')
# ax1.scatter(x[:10,0].cpu(), x[:10,1].cpu(), label="$X$ Samples", marker="o", facecolor="r", edgecolor="k")
# ax1.legend()
#
# CS2 = ax2.contourf(x1, x2, qy_grid,100, cmap=plt.cm.YlGnBu)
# ax2.set_title("Distribution of $Y \sim Q(Y)$")
# ax2.set_xlabel('$y_1$')
# ax2.set_ylabel('$y_2$')
# ax2.set_aspect('equal')
# ax2.scatter(y[:10,0].cpu(), y[:10,1].cpu(), label="$Y$ Samples", marker="o", facecolor="r", edgecolor="k")
# ax2.legend()
# #ax1.axis([-2.5, 2.5, -2.5, 2.5])
#
# # Add colorbar and title
# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
# cbar = fig.colorbar(CS2, cax=cbar_ax)
# cbar.ax.set_ylabel('Density results')
# plt.suptitle("MMD result: {round(result.item(),3)}",y=0.95, fontweight="bold")
# plt.savefig('MMD.png')