import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_classification
from sklearn.svm import SVC

# Generate synthetic data for binary classification
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
y = 2 * y - 1  # Convert labels to -1 and 1

# SVM initialization
svc = SVC(kernel='linear', C=1, max_iter=1, tol=1e-3)

# Prepare the figure
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
line, = ax.plot([], [], 'k-', lw=2, label='Hyperplane')
margin_up, = ax.plot([], [], 'k--', lw=1, label='Margin')
margin_down, = ax.plot([], [], 'k--', lw=1)
ax.legend()
ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
ax.set_title("SVM Hyperplane Optimization")

# Animation function
def update(frame):
    svc.max_iter = frame + 1
    svc.fit(X, y)
    coef = svc.coef_.ravel()
    intercept = svc.intercept_

    # Calculate hyperplane and margins
    x_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    y_vals = -(coef[0] * x_vals + intercept) / coef[1]
    margin1 = y_vals - 1 / coef[1]
    margin2 = y_vals + 1 / coef[1]

    # Update plot
    line.set_data(x_vals, y_vals)
    margin_up.set_data(x_vals, margin1)
    margin_down.set_data(x_vals, margin2)

# Create animation
ani = FuncAnimation(fig, update, frames=100, repeat=False)
plt.show()
