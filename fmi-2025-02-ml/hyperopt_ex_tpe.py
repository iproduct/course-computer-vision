# TPE = Tree-structured Parzen Estimator.

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

# Load data
X, y = load_iris(return_X_y=True)

# Define objective function
def objective(params):
    clf = RandomForestClassifier(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        min_samples_split=int(params['min_samples_split']),
        random_state=42
    )
    score = cross_val_score(clf, X, y, cv=3).mean()
    return {'loss': -score, 'status': STATUS_OK}  # minimize negative accuracy

# Define search space
space = {
    'n_estimators': hp.quniform('n_estimators', 50, 300, 1),
    'max_depth': hp.quniform('max_depth', 3, 15, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1)
}

# Run TPE optimization
trials = Trials()
best_params = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,  # TPE sampler
    max_evals=50,
    trials=trials,
    rstate=np.random.default_rng(42)
)

print("Best hyperparameters:", best_params)

best_model = RandomForestClassifier(
        n_estimators=int(best_params['n_estimators']),
        max_depth=int(best_params['max_depth']),
        min_samples_split=int(best_params['min_samples_split']),
    )
scores = cross_val_score(best_model, X, y, cv=5)  # 5-fold CV for evaluation

print("\nTesting results with best hyperparameters:")
print(f"Mean Accuracy: {scores.mean():.4f}")
print(f"Accuracy Std Dev: {scores.std():.4f}")
print(f"All fold accuracies: {scores}")


# ------------------------------
# Plotting results
# ------------------------------
# Extract losses from trials
losses = [trial['result']['loss'] for trial in trials.trials]
evaluations = range(1, len(losses)+1)

plt.figure(figsize=(10, 6))
plt.plot(evaluations, losses, marker='o', linestyle='-', color='blue')
plt.title("TPE Optimization: Loss over Trials")
plt.xlabel("Trial")
plt.ylabel("Loss (negative accuracy)")
plt.grid(True)
plt.show()