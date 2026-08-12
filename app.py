import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns


st.title("Number of Orders Prediction")

df = pd.read_csv("order_pred.csv")

df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Order_Day'] = df['Order Date'].dt.date


# AGGREGATION (THIS IS THE KEY FIX)
df = df.groupby(
    ['Order_Day', 'Product'],
    as_index=False
).agg({
    'Quantity Ordered': 'sum',
    'Price Each': 'mean'
})


# NOW show preview (multiple products will appear)
st.subheader("Dataset Preview (Aggregated)")
st.dataframe(df.head(10))


df['Order_Day'] = pd.to_datetime(df['Order_Day'])
df['day'] = df['Order_Day'].dt.day
df['month'] = df['Order_Day'].dt.month
df['weekday'] = df['Order_Day'].dt.weekday

df = df.drop(columns=['Order_Day'])


le = LabelEncoder()
df['Product'] = le.fit_transform(df['Product'])


X = df.drop(columns=['Quantity Ordered'])
y = df['Quantity Ordered']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"MAE: {mae:.2f}")
st.write(f"RMSE: {rmse:.2f}")
st.write(f"R2 Score: {r2:.2f}")


#EVERYTHING BELOW REMAINS EXACTLY THE SAME 

st.subheader("Orders Over Time")
fig1, ax1 = plt.subplots()
sns.lineplot(x=df.index, y=df['Quantity Ordered'], ax=ax1)
ax1.set_xlabel("Index (Time)")
ax1.set_ylabel("Quantity Ordered")
st.pyplot(fig1)


st.subheader("Average Orders per Product")
fig2, ax2 = plt.subplots()
sns.barplot(
    x='Product',
    y='Quantity Ordered',
    data=df,
    estimator='mean',
    ax=ax2
)
ax2.set_xlabel("Product")
ax2.set_ylabel("Average Quantity Ordered")
st.pyplot(fig2)


st.subheader("Actual vs Predicted Orders")
fig3, ax3 = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax3)
ax3.set_xlabel("Actual Orders")
ax3.set_ylabel("Predicted Orders")
st.pyplot(fig3)


st.subheader("Actual vs Predicted Orders (Sample)")
fig4, ax4 = plt.subplots()
ax4.plot(y_test.values[:50], label="Actual Orders")
ax4.plot(y_pred[:50], label="Predicted Orders")
ax4.set_xlabel("Sample Index")
ax4.set_ylabel("Orders")
ax4.legend()
st.pyplot(fig4)
