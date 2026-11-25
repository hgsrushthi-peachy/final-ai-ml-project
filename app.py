# House Price Prediction - Final AI & ML Project
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Sample data - Area, Bedrooms, Age -> Price
data = {
    'Area': [1000, 1500, 2000, 2500, 1200, 1800, 2200, 1600, 3000, 2800],
    'Bedrooms': [2, 3, 3, 4, 2, 3, 4, 3, 5, 4],
    'Age': [10, 5, 2, 1, 8, 4, 3, 6, 1, 2],
    'Price': [50, 75, 95, 120, 58, 85, 105, 78, 150, 135]
}
df = pd.DataFrame(data)

X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Accuracy (R2 Score):", r2_score(y_test, model.predict(X_test)))

# Prediction for new house
new_house = [[2000, 3, 5]]
price = model.predict(new_house)
print(f"2000 sqft, 3 Bedroom, 5 saal purane ghar ki price: {price[0]:.2f} Lakh")
