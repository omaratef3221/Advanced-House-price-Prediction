from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from data import get_cleaned_dataset
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error

def train():
    data = get_cleaned_dataset()
    
    Y = data["PRICE"]
    X = data.drop(["PRICE"], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=10)
    
    regressor = RandomForestRegressor(random_state=10)
    regressor.fit(X_train, y_train)
    
    
    predictions = regressor.predict(X_test)
    
    print("MSE: ", mean_squared_error(y_test, predictions))
    print("RMSE: ", root_mean_squared_error(y_test, predictions))
    print("R2_Score: ", r2_score(predictions, y_test))
    return regressor

