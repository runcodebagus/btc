
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("btc_price")


#using required features
cdf = df[['price_open','price_high','price_low','price_close']]


#Training Data and Predictor Variable

x = cdf.iloc[:, :3]
y = cdf.iloc[:, -1]


regressor = LinearRegression()


regressor.fit(x, y)


pickle.dump(regressor, open('model.pkl','wb'))

