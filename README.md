## Normal Linear Regression

This repository contains an implementation of Normal Linear Regression in Python. Normal Linear Regression is a supervised learning algorithm used for predicting numeric values based on a linear relationship between the input features and the target variable.

### Class: NormalLinearRegression

The `NormalLinearRegression` class provides a simple and efficient implementation of Normal Linear Regression. It includes the following methods:

#### `__init__(self)`

The constructor initializes the attributes of the class:

- `X`: Input/feature matrix.
- `Y`: Target matrix.
- `theta`: Array of the optimal values of weights.

#### `fit(self, x, y)`

This method trains the Normal Linear Regression model on the provided input data `x` and target values `y`. It returns the optimal values of weights (theta) that best fit the data.

- `x`: Input/feature matrix.
- `y`: Target matrix.

#### `predict(self, x)`

The `predict()` method makes predictions using the trained model on new input data `x`. It returns the predicted target values based on the learned weights (theta).

- `x`: Test input/feature matrix.

#### `calculate_theta(self)`

This internal method is used by the `fit()` method to calculate the optimal values of weights (theta) using the Normal Equation. The Normal Equation allows finding the weights directly without using an iterative optimization algorithm like gradient descent.

### Usage

```python
# Import the necessary libraries
import numpy as np

# Create a NormalLinearRegression object
model = NormalLinearRegression()

# Prepare your training data - x and y are NumPy arrays with input features and target values
x = ...
y = ...

# Train the model and obtain the optimal weights
optimal_weights = model.fit(x, y)

# Make predictions on new data
new_data = ...
predictions = model.predict(new_data)
```

### Example

Here's a simple example using the NormalLinearRegression class:

```python
import numpy as np

# Create sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Create a NormalLinearRegression object
model = NormalLinearRegression()

# Train the model and obtain the optimal weights
optimal_weights = model.fit(x, y)

# Make predictions on new data
new_data = np.array([6, 7, 8])
predictions = model.predict(new_data)

print("Optimal Weights:", optimal_weights)
print("Predictions:", predictions)
```

### Contribution

Contributions to this repository are welcome! If you find any bugs, have feature requests, or want to optimize the code, feel free to submit issues or pull requests.

### Dependencies

The implementation relies on NumPy for efficient numerical computations. Ensure you have NumPy installed before using the code:

```
pip install numpy
```

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Explore the world of Normal Linear Regression and build powerful predictive models! Happy coding!
