import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("prediction.csv")

columns = ['superficie', 'chambres', 'Proximite', 'Prix']
df = df[columns]

X = df.iloc[:, 0:3]
y = df.iloc[:, 3:]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LinearRegression()
lr.fit(X_train, y_train)

pickle.dump(lr, open('model1.pkl', 'wb'))
