import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_csv('kc_house_data.csv')

print(df.head()) #display first 5 rows

print(df.info()) # Type of data that is used
 
print(df.describe()) #5 summary stat
print(df.isnull().sum())


df = df.drop(["id", "date"], axis=1) #drop the columns that prove little correlation to the price of the home

#create target and feature variables, Take the target out of the dataframe
X = df.drop("price", axis=1) #features
y = df["price"] #target


#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R^2: {r2}")

rf =  RandomForestRegressor()
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)

print("Random Forest RMSE:",mean_squared_error(y_test, rf_pred))
print("Random Forest R^2:", r2_score(y_test, rf_pred))


plt.scatter(y_test, rf_pred, alpha=0.3)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices (Random Forest)")
plt.grid(True)
plt.show()

# Get feature importances from the trained Random Forest model
importances = rf.feature_importances_
features = X.columns

# Create a DataFrame to organize and sort them
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Plot a horizontal bar chart
sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("Feature Importance from Random Forest")
plt.tight_layout()
plt.show()


#Feature Engineering the data
df_fe =  pd.read_csv('kc_house_data.csv')

#Using Time
df_fe["date"] = pd.to_datetime(df_fe["date"])
df_fe["year"] = df_fe["date"].dt.year
df_fe["month"] = df_fe["date"].dt.month
df_fe["house_age"] = df_fe["year"] - df_fe["yr_built"]
df_fe.drop(columns=["date"], inplace=True)
df_fe["price_per_sqft"] = df_fe["price"] / df_fe["sqft_living"]

df_fe["waterfront"] = df_fe["waterfront"].apply(lambda x: 1 if x == 1 else 0)
df_fe = pd.get_dummies(df_fe, columns=["view", "condition"], drop_first=True)

#combining different features 
df_fe["bed_bath_rooms"] = df_fe["bedrooms"] + df_fe["bathrooms"]
df_fe["sqft_total"] = df_fe["sqft_living"] + df_fe["sqft_basement"]
df_fe["bedrooms_per_room"] = df_fe["bedrooms"] / (df_fe["sqft_living"] + 1)

#converting to log scale
df_fe["log_price"] = np.log1p(df_fe["price"])
df_fe["log_sqft_living"] = np.log1p(df_fe["sqft_living"])
#converting into specific categories 
df_fe["grade_cat"] = pd.cut(df_fe["grade"], bins=[0, 5, 8, 13], labels=["low", "medium", "high"])
df_fe = pd.get_dummies(df_fe, columns=["grade_cat"], drop_first=True)
#Bin square footage
df_fe["sqft_bin"] = pd.qcut(df_fe["sqft_living"], q=4, labels=False)
#getting rid of useless features
df_fe.drop(columns=["id",  "price", "sqft_living", "yr_built"], inplace=True)

#Scaling to make the numbers better
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_fe.drop("log_price", axis=1))
X = df_fe.drop("log_price", axis=1)
y = df_fe["log_price"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf = RandomForestRegressor()
rf.fit(X_train, y_train)


preds = rf.predict(X_test)
rmse = mean_squared_error(y_test, preds) ** 0.5
r2 = r2_score(y_test, preds)

print("RMSE:", rmse)
print("R²:", r2)
