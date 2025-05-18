# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and prepare data:
   Load the California housing dataset and split it into features X and targets y (AveOccup, Housing_price).
2. Split and scale:
   Split into train/test sets and apply StandardScaler to normalize the input features.
3. Train model:
   Use SGDRegressor inside MultiOutputRegressor to train on scaled features and multi-target outputs.
4. Predict and evaluate:
   Predict on test data and compute mean squared error for both targets to assess performance.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Arunsamy D
RegisterNumber:  212224240016 / 24900591
*/
```

```
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
```

```
data = pd.DataFrame(  fetch_california_housing().data,                        
                      columns=fetch_california_housing().feature_names,
)
data.head()
```

```
data['Housing_price']= fetch_california_housing().target
data.head()
```

```
x = data.drop(columns=["AveOccup", "Housing_price"])
y = data[["AveOccup", "Housing_price"]]

print("Input Features \n\n",x)
print("\nPredicting Features\n\n",y)
```

```
x_train, x_test, y_tain, y_test = train_test_split( 
                                                    x,
                                                    y,
                                                    test_size = 0.2,     
                                                    random_state = 42,
                                                  )

```

```
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
```

```
model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)

multi_output_model = MultiOutputRegressor(model)
multi_output_model.fit(x_train_scaled, y_tain)
```

```
y_predict = multi_output_model.predict(x_test_scaled)
y_predict
```

```
mse = mean_squared_error(y_test, y_predict, multioutput='raw_values')
print(f"Mean Squared Error for Median House Value: {mse[0]:.4f}")
print(f"Mean Squared Error for Average Occupants: {mse[1]:.4f}")
```
## Output:

![image](https://github.com/user-attachments/assets/c84655d4-3ad6-4c10-9efe-a66b6c572f2c)

![image](https://github.com/user-attachments/assets/5744c211-ac9d-4b46-9b27-54e3eff9416a)

![image](https://github.com/user-attachments/assets/1677c9f1-f500-451a-a9eb-50f40482e504)

![image](https://github.com/user-attachments/assets/48afc62e-93ef-4d58-921c-99de6be0fd58)

![image](https://github.com/user-attachments/assets/1ce3f3c1-4075-4149-92f5-92cb659f082c)

![image](https://github.com/user-attachments/assets/5ab4a7b5-7e95-47c2-804c-dbc36837f083)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
