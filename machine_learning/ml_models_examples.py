# ML Models: runnable core examples
# Requires numpy, scikit-learn; run top to bottom
# Templates and optional XGBoost are intentionally excluded.

import numpy as np
from sklearn.datasets import make_regression, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score

# Synthetic data: 600 rows and 4 features; no download needed.
Xr, yr = make_regression(n_samples=600, n_features=4,
                         noise=15, random_state=42)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.2, random_state=42)
Xc, yc = load_breast_cancer(return_X_y=True)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.2, stratify=yc, random_state=42)

def report_reg(model):
    model.fit(Xr_train, yr_train)
    pred = model.predict(Xr_test)
    print(type(model).__name__, "RMSE =",
          np.sqrt(mean_squared_error(yr_test, pred)))

def report_clf(model):
    model.fit(Xc_train, yc_train)
    print(type(model).__name__, "accuracy =",
          accuracy_score(yc_test, model.predict(Xc_test)))

from sklearn.linear_model import LinearRegression
lin = LinearRegression()
report_reg(lin)
print("b:", lin.intercept_, "w:", lin.coef_)
# Verify that prediction equals the dot product plus the intercept.
assert np.allclose(lin.predict(Xr_test),
                   Xr_test @ lin.coef_ + lin.intercept_)

from sklearn.preprocessing import PolynomialFeatures
poly = make_pipeline(PolynomialFeatures(degree=2, include_bias=False),
                     StandardScaler(), LinearRegression())
report_reg(poly)

# Batch gradient descent from scratch; scaling helps convergence.
sc = StandardScaler()
Xs = sc.fit_transform(Xr_train)
Xb = np.c_[np.ones(len(Xs)), Xs]
theta = np.zeros(Xb.shape[1])
for step in range(1000):
    gradient = Xb.T @ (Xb @ theta - yr_train) / len(Xb)
    theta -= 0.05 * gradient
print("Batch GD theta:", theta)

from sklearn.linear_model import SGDRegressor
online = SGDRegressor(max_iter=3000, tol=1e-4, random_state=42)
online.partial_fit(Xs[:64], yr_train[:64])
online.partial_fit(Xs[64:128], yr_train[64:128])
# Demonstrate incremental learning; this model is not fully trained, so do not compare its result with other models.

from sklearn.linear_model import Ridge, Lasso, ElasticNet
for model in [Ridge(alpha=10), Lasso(alpha=0.5),
              ElasticNet(alpha=0.5, l1_ratio=0.3)]:
    report_reg(make_pipeline(StandardScaler(), model))

from sklearn.linear_model import LogisticRegression
logit = make_pipeline(StandardScaler(),
                      LogisticRegression(C=1.0, max_iter=2000))
report_clf(logit)
print("class order:", logit.classes_)
print("probability:", logit.predict_proba(Xc_test[:2]))
# For multiclass classification, use the same model with more than two classes.
from sklearn.datasets import load_iris
Xi, yi = load_iris(return_X_y=True)
softmax = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
softmax.fit(Xi, yi)
print(softmax.predict_proba(Xi[:1]))

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
report_clf(LinearDiscriminantAnalysis())
report_clf(QuadraticDiscriminantAnalysis(reg_param=0.1))

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.tree import export_text
tr = DecisionTreeRegressor(max_depth=3, min_samples_leaf=10, random_state=42)
report_reg(tr)
print(export_text(tr, feature_names=["x1", "x2", "x3", "x4"]))
report_clf(DecisionTreeClassifier(max_depth=3, random_state=42))

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=150, max_features=0.8,
                           min_samples_leaf=3, random_state=42, n_jobs=-1)
report_reg(rf)
per_tree = np.array([t.predict(Xr_test[:1])[0] for t in rf.estimators_])
assert np.allclose(per_tree.mean(), rf.predict(Xr_test[:1])[0])
print("mean tree prediction:", per_tree.mean())

from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
                         n_estimators=100, learning_rate=0.5,
                         random_state=42)
report_clf(ada)
print("first model weights:", ada.estimator_weights_[:5])
report_reg(AdaBoostRegressor(n_estimators=100, random_state=42))

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
# Implement the core of squared-loss boosting from scratch.
base = yr_train.mean()
train_pred = np.full(len(yr_train), base)
new_pred = np.full(len(Xr_test), base)
trees = []
for t in range(50):
    residual = yr_train - train_pred  # Pseudo-residual for half squared loss.
    tree = DecisionTreeRegressor(max_depth=2, random_state=t)
    tree.fit(Xr_train, residual)
    train_pred += 0.1 * tree.predict(Xr_train)
    new_pred += 0.1 * tree.predict(Xr_test)
    trees.append(tree)
print("manual GB RMSE:", np.sqrt(mean_squared_error(yr_test, new_pred)))
report_reg(GradientBoostingRegressor(n_estimators=150, learning_rate=0.05,
                                    max_depth=2, random_state=42))

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
report_reg(make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7)))
report_clf(make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)))

from sklearn.svm import SVC, SVR
report_clf(make_pipeline(StandardScaler(), SVC(C=1, kernel="rbf", gamma="scale")))
report_reg(make_pipeline(StandardScaler(), SVR(C=100, epsilon=5)))

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
report_clf(nb)
print("means shape:", nb.theta_.shape)
print("variances shape:", nb.var_.shape)

from sklearn.neural_network import MLPClassifier
mlp = make_pipeline(StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu",
                  early_stopping=True, max_iter=400, random_state=42))
report_clf(mlp)
print("weight matrix shapes:", [w.shape for w in mlp[-1].coefs_])

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
Xu, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)
Xu = StandardScaler().fit_transform(Xu)
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Xu)
db = DBSCAN(eps=0.25, min_samples=5).fit(Xu)
gm = GaussianMixture(n_components=3, n_init=5, random_state=42).fit(Xu)
print("centroids:", km.cluster_centers_)
print("DBSCAN labels:", np.unique(db.labels_))
print("GMM responsibilities:", gm.predict_proba(Xu[:2]))

from sklearn.decomposition import PCA
pca_pipe = make_pipeline(StandardScaler(), PCA(n_components=0.95))
Z = pca_pipe.fit_transform(Xc_train)
print("before/after:", Xc_train.shape, Z.shape)
print("retained variance:",
      pca_pipe[-1].explained_variance_ratio_.sum())
# If applying PCA before classification, put both in a Pipeline before cross-validation.

# Example of one transition, not a complete reinforcement learning training loop.
Q = np.zeros((3, 2))  # 3 states, 2 actions
s, a, reward, next_s, done = 0, 1, 1.0, 2, True
alpha, discount = 0.1, 0.95
bootstrap = 0.0 if done else discount * Q[next_s].max()
Q[s, a] += alpha * (reward + bootstrap - Q[s, a])
print("updated Q:", Q[s, a])

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
pipe = make_pipeline(StandardScaler(), Ridge())
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, Xr_train, yr_train, cv=cv,
                         scoring="neg_root_mean_squared_error")
print("CV RMSE by fold:", -scores)
print("mean/std:", (-scores).mean(), (-scores).std())
search = GridSearchCV(pipe, {"ridge__alpha": [0.1, 1, 10, 100]},
                       cv=cv, scoring="neg_root_mean_squared_error",
                       refit=True)
search.fit(Xr_train, yr_train)
print("best hyperparameter:", search.best_params_)
print("selected CV RMSE:", -search.best_score_)
print("final holdout RMSE:", np.sqrt(mean_squared_error(
    yr_test, search.predict(Xr_test))))
