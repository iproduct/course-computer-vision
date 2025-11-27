#  data_url = "http://lib.stat.cmu.edu/datasets/boston"
#     raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
#     data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
#     target = raw_df.values[1::2, 2]
#
# Alternative datasets include the California housing dataset and the
# Ames housing dataset. You can load the datasets as follows::
#
#     from sklearn.datasets import fetch_california_housing
#     housing = fetch_california_housing()
#
# for the California housing dataset and::
#
#     from sklearn.datasets import fetch_openml
#     housing = fetch_openml(name="house_prices", as_frame=True)
#
# for the Ames housing dataset.

# Load a small sample dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load California housing dataset as pandas DataFrame
housing = fetch_california_housing(as_frame=True)

X = housing.data
y = housing.target

pd.set_option('display.max_columns', None)
print(X.describe())
print(X.shape, y.shape)
print(X.head(10, ), y.head(10, ))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Train the model
gbr.fit(X_train, y_train)

# Make predictions
y_pred = gbr.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.3f}")

# Calculate test set deviance for each boosting iteration
test_score = np.zeros((gbr.n_estimators,), dtype=np.float64)
for i, y_pred_iter in enumerate(gbr.staged_predict(X_test)):
    test_score[i] = mean_squared_error(y_test, y_pred_iter)

# Plot training and test deviance
plt.figure(figsize=(8, 6))
plt.plot(np.arange(gbr.n_estimators) + 1, gbr.train_score_, 'b-', label='Training Set Deviance')
plt.plot(np.arange(gbr.n_estimators) + 1, test_score, 'r-', label='Test Set Deviance')
plt.title('Gradient Boosting Deviance over Boosting Iterations (California Housing)')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance (Mean Squared Error)')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# Scatter plot of two most predictive features vs house value (target)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X['MedInc'], X['AveRooms'], c=y, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Median House Value (100k $)')
plt.xlabel('Median Income (MedInc)')
plt.ylabel('Average Rooms (AveRooms)')
plt.title('Scatter Plot: Median Income vs Average Rooms Colored by House Value')
plt.tight_layout()
plt.show()