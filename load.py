import xgboost as xgb
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np

my_imputer = SimpleImputer()

df = pd.read_csv('autos.csv')
features = ['stroke', 'bore', 'width', 'length', 'height', 'wheel_base']

y = df.price

X = df[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=3)

print(val_X.iloc[0])
print(len(X.iloc[0]))
print(type(val_X.iloc[0]))
train_y = pd.DataFrame(my_imputer.fit_transform(train_y.values.reshape(-1, 1)))
val_y = pd.DataFrame(my_imputer.transform(val_y.values.reshape(-1, 1)))

loaded_model = xgb.XGBRegressor()
loaded_model.load_model('model.json')

evaluation = [(train_X, train_y), (val_X, val_y)] 

loaded_model.fit(train_X, train_y, eval_set=evaluation, verbose=2)

# Create DMatrix from the selected row of the DataFrame, enabling categorical data

# Use the loaded model to make a prediction
#pred = loaded_model.predict(xgb.DMatrix(x))
#pred = loaded_model.predict(val_X.loc[2, :].to_numpy().reshape(-1, 1))
pred = loaded_model.predict(val_X.loc[1, :])
print(pred)
