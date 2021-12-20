import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from gpdyn.models import GpDynModel
from gpdyn.tests.ground_truths import SimpleAr_2D

f = SimpleAr_2D()
y_train = f.get_data(y_init=10., t_sim=30, noise_sigma=0.2)
y_test = f.get_data(y_init=20., t_sim=20, noise_sigma=0.2)

model = GpDynModel()
model.load_data(y_train)
model.train_model(delays=2)
_, y_true, y_mean, _ = model.test_model(y_test.reshape(1,-1), horizon=1)

density = 20
y1 = np.linspace(0., 10., density)
y2 = np.linspace(0., 10., density)
yy1, yy2 = np.meshgrid(y1, y2)

ground_truth = f(np.stack([yy1.flatten(), yy2.flatten()], axis=0))

feats = np.stack([yy1.flatten(),yy2.flatten()], axis=1)
pred, _ = model.gp.predict_f(feats)
pred = pred.numpy()

# Time-series results
fig, ax = plt.subplots(2, 1)
ax[0].plot(y_train, '-ko')
ax[0].set_title("Training data")
ax[0].grid(visible=True)
ax[1].plot(y_true[0,:], '--ko')
ax[1].plot(y_mean, '-bd')
ax[1].set_title("Test dataset")
ax[1].grid(visible=True)
ax[1].legend(["Ground-truth", "Predictions"], loc='upper right', ncol=1, framealpha=0.5)
plt.draw()
plt.pause(.001)

# Surface results
axs = []
fig = plt.figure()
axs.append(fig.add_subplot(121, projection='3d'))
axs.append(fig.add_subplot(122, projection='3d'))
color = cm.RdYlGn
axs[0].plot_surface(yy1, yy2, ground_truth.reshape(yy1.shape), cmap=color, linewidth=0, antialiased=False, alpha=0.5)
axs[1].plot_surface(yy1, yy2, pred.reshape(yy1.shape), cmap=color, linewidth=0, antialiased=False, alpha=0.5)
axs[0].set_title("Ground-truth")
axs[1].set_title("Gaussian process")
for ax in axs:
    ax.scatter(y_train[1:-1], y_train[:-2], y_train[2:], s=30, c='k', marker='o')
    ax.set_xlim(0,10)
    ax.set_ylim(0,10)
    ax.set_zlim(0,10)
    ax.set_xlabel("y[t]")
    ax.set_ylabel("y[t-1]")
    ax.set_zlabel("y[t+1]")
plt.draw()
plt.pause(.001)

input("\nPress any key to terminate...")
