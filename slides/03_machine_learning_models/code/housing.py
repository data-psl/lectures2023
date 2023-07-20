#!/usr/bin/env python3

from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np

california_housing = fetch_california_housing(as_frame=True)


X = california_housing.data
Y = california_housing.target


print(X.columns)

x = X["MedInc"].to_numpy()
y = Y.to_numpy()
I = y < 4.5
x, y = x[I], y[I]

rc = {
    "font.size": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
}
plt.rcParams.update(rc)


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

i_train, i_test = train_test_split(np.arange(len(x)), test_size=0.5, random_state=0)
x_train = x[i_train]
x_test = x[i_test]
y_train = y[i_train]
y_test = y[i_test]
algo = Pipeline(
    [
        ("poly", PolynomialFeatures(degree=3)),
        ("linear", LinearRegression()),
    ]
)
algo.fit(x_train.reshape(-1, 1), y_train)

y_pred = algo.predict(x_test.reshape(-1, 1))

plt.figure(figsize=(6, 3))
plt.scatter(x_test[0], y_test[0], marker="*", alpha=1)
plt.xlabel("Median income")
plt.xlim([0, 10])
plt.ylim([0, 4.6])
plt.ylabel("Median price")
plt.savefig("../images/california_1.png", bbox_inches="tight")

plt.figure(figsize=(6, 3))
plt.scatter(x_test, y_test, marker="*", alpha=0.1)
plt.xlabel("Median income")
plt.xlim([0, 10])
plt.ylim([0, 4.6])
plt.ylabel("Median price")
plt.savefig("../images/california_2.png", bbox_inches="tight")

plt.figure(figsize=(6, 3))
plt.scatter(x_test, y_test, marker="*", alpha=0.1, vmin=1, vmax=3)
ixtest = np.argsort(x_test)
plt.plot(x_test[ixtest], y_pred[ixtest], color="red", label="pred")
plt.xlabel("Median income")
plt.xlim([0, 10])
plt.ylim([0, 4.6])
plt.legend()
plt.ylabel("Median price")
plt.savefig("../images/california_3.png", bbox_inches="tight")


# %%

x1 = X["Latitude"].to_numpy()
x2 = X["Longitude"].to_numpy()
y = Y.to_numpy()

plt.figure(figsize=(6, 3))
plt.scatter(x1[I][i_test], x2[I][i_test], c=y[I][i_test], alpha=0.5, vmin=1, vmax=3)
plt.colorbar()
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.xlim([32.5, 41])
plt.ylim([-114, -123])
plt.savefig("../images/california_4.png", bbox_inches="tight")

# %%

x = np.array([x1, x2]).T

x_test = x[I][i_test]
x_train = x[I][i_train]

algo = DecisionTreeRegressor(max_depth=4)
algo.fit(x_train, y_train)

gx1 = np.linspace(32.5, 41, 200)
gx2 = np.linspace(-114, -123, 200)

gplotx = []
gploty = []
gplotc = []
for a in gx1:
    for b in gx2:
        gplotx.append(a)
        gploty.append(b)
        z = algo.predict([[a, b]])[0]
        gplotc.append(z)

print(np.max(gplotc), np.min(gplotc))

plt.figure(figsize=(6, 3))
plt.scatter(gplotx, gploty, c=gplotc, alpha=0.02, vmin=1, vmax=3)
plt.scatter(x1[I][i_test], x2[I][i_test], c=y[I][i_test], alpha=0.5, vmin=1, vmax=3)
plt.colorbar()
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.xlim([32.5, 41])
plt.ylim([-114, -123])
plt.savefig("../images/california_5.png", bbox_inches="tight")
