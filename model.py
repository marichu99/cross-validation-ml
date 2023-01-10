from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd


cols_to_use=["Rooms","Distance","Landsize","BuildingArea","YearBuilt"]
data = pd.read_csv("melb_data.csv")
y=data.Price
x=data[cols_to_use]

# perform splits on data 
X_train,X_valid,y_train,y_valid=train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=0)

# create a pipeline
pipe_line=Pipeline(steps=[
    ("impute",SimpleImputer()),
    ("model",RandomForestRegressor(n_estimators=100,random_state=0))
])

# perform cross-validation
scores = -1 * cross_val_score(pipe_line,X=x,y=y,cv=5,scoring="neg_mean_absolute_error")
print("MAE scores are:\n",scores)
print("Average mean absolute error is \n",scores.mean())
