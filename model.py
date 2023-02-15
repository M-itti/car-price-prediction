#!/usr/bin/env python3.10
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

import numpy as np
import pandas as pd
import mlflow
import xgboost as xg


# initializations
features = ['stroke', 'bore', 'width', 'length', 'height', 'wheel_base']
DF = pd.read_csv('autos.csv')
my_imputer = SimpleImputer()


y = DF.price
X = DF[features]

for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=3)

train_y = pd.DataFrame(my_imputer.fit_transform(train_y.values.reshape(-1, 1)))
val_y = pd.DataFrame(my_imputer.transform(val_y.values.reshape(-1, 1)))

'''
iowa_model = xg.XGBRegressor(max_depth=10, learning_rate=0.01, n_estimators=500, objective='reg:squarederror', booster='gbtree', verbosity=2, tree_method='gpu_hist', gpu_id=0)

iowa_model.fit(train_X, train_y) 

# validation prediction prediction
val_predictions = iowa_model.predict(val_X)

# calculate RMSE
from sklearn.metrics import mean_squared_error
res = mean_squared_error(val_y, val_predictions, squared=False)
print(f"Validation RMSE : {res}")

'''

# **** Hyperparameter-tuning ****

from hyperopt import fmin, Trials, hp, tpe, STATUS_OK 
from hyperopt.pyll.base import scope
import xgboost 

#Define the space over which hyperopt will search for optimal hyperparameters.
space = {'max_depth': scope.int(hp.quniform("max_depth", 1, 5, 1)),
        'gamma': hp.uniform ('gamma', 0,1),
        'reg_alpha' : hp.uniform('reg_alpha', 0,50),
        'reg_lambda' : hp.uniform('reg_lambda', 10,100),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0,1),
        'min_child_weight' : hp.uniform('min_child_weight', 0, 5),
        'n_estimators': 10000,
        'learning_rate': hp.uniform('learning_rate', 0, .15),
        'tree_method':'gpu_hist', 
        'gpu_id': 0,
        'random_state': 5,
        'max_bin' : scope.int(hp.quniform('max_bin', 200, 550, 1))}

# Best model parameters
space = {'max_depth': 5,
        'gamma': 0.9957775271841635,
        'reg_alpha' : 19.207644334896973,
        'reg_lambda' : 89.65684562378941,
        'colsample_bytree' : 0.5224127388761285,
        'min_child_weight' : 0,
        'learning_rate': 0.08418802162536007,
        'tree_method':'gpu_hist', 
        'gpu_id': 0,
        'random_state': 5,
        'max_bin' : 204}

#Define the hyperopt objective.
def hyperparameter_tuning(space):
    # Log the XGboost model with mlflow
    with mlflow.start_run():
        evaluation = [(train_X, train_y), (val_X, val_y)]
        model = xgboost.XGBRegressor(**space, early_stopping_rounds=1000, objective='reg:squarederror', n_estimators=10000)
        model.set_params(eval_metric='rmse')
        model.fit(train_X, train_y, eval_set=evaluation, verbose=2)
        pred = model.predict(val_X)
        rmse = mean_squared_error(val_y, pred, squared=False)
        print ("SCORE:", rmse)

        mlflow.log_params(space)
        mlflow.log_metric('score', rmse)

        mlflow.xgboost.log_model(model, 'model')
    
    # Save the model to disk
    model.save_model("model.json")

    #Specify what the loss is for each model.
    return {'loss':rmse, 'status': STATUS_OK, 'model': model}

res = hyperparameter_tuning(space)
print(res)

'''
#Run 20 trials.
trials = Trials()
best = fmin(fn=hyperparameter_tuning,
            space=space,
            algo=tpe.suggest,
            max_evals=30,
            trials=trials)
print(best, end='\n')

#Create instace of best model.
best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']

#Examine model hyperparameters
print(best_model)

# Examine RMSE using Optimal hyperparameter vs Standard Hyperparameters
xgb_preds_best = best_model.predict(val_X)
xgb_score_best = mean_squared_error(val_y, xgb_preds_best, squared=False)
print('RMSE_Best_Model:', xgb_score_best)
'''


# **** Cross-validation **** # 

train_X = xg.DMatrix(train_X, label=train_y)

xgboost.cv(space, train_X, num_boost_round=2, nfold=5,
       metrics={'error'}, seed=0,
       callbacks=[xgboost.callback.EvaluationMonitor(show_stdv=True)],
       verbose_eval=2)

print('running cross validation, disable standard deviation display')
# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value
res = xgboost.cv(space, train_X, num_boost_round=10, nfold=5,
             metrics={'error'}, seed=0,
             callbacks=[xgboost.callback.EvaluationMonitor(show_stdv=False),
                        xgboost.callback.EarlyStopping(3)])
print(res)
