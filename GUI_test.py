# encoding=utf-8
import numpy as np
from sklearn.preprocessing import scale, StandardScaler

a = np.array([[1.0, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
a_ = scale(a)
print(a_)
X_scaled = StandardScaler().fit(a)
print(X_scaled)

print(a_*X_scaled.scale_ + X_scaled.mean_)
