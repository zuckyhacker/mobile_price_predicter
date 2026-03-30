import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

# load dataset
data = pd.read_csv("mobile_data.csv")

# convert price
if 'price_range' in data.columns:
    data['price'] = data['price_range'] * 10000

# encode brand
le = LabelEncoder()
data['brand'] = le.fit_transform(data['brand'])

# features and target
X = data.drop(['price', 'price_range'], axis=1, errors='ignore')
y = data['price']

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model
model = RandomForestRegressor(n_estimators=200)
model.fit(X_train, y_train)

# accuracy
print("Accuracy:", model.score(X_test, y_test))

# save files
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns, open("features.pkl", "wb"))
pickle.dump(le, open("brand_encoder.pkl", "wb"))
