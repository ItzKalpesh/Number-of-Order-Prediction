# Number of Orders Prediction

## 📌 Overview

This project focuses on predicting the number of customer orders using historical sales and order data. The aim is to understand order patterns and use machine learning to estimate order demand, which can support business decisions such as inventory planning, logistics, and resource management.

## 🎯 Objective

To build a machine learning model that predicts the number of orders based on historical order information such as date, product, quantity ordered, and price.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit

## 🔄 Project Workflow

1. Load the historical order dataset.
2. Clean and preprocess the data.
3. Convert order dates into usable date features.
4. Aggregate orders by product and day.
5. Encode categorical product data.
6. Split the data into training and testing sets.
7. Train a Linear Regression model.
8. Generate order predictions.
9. Evaluate the model using MAE, RMSE, and R².
10. Visualize order trends and actual vs. predicted values.
11. Display the results in a browser using Streamlit.

## 📊 Visualizations

The Streamlit application displays:

* Orders over time
* Average orders per product
* Actual vs. predicted orders using a scatter plot
* Actual vs. predicted orders using a line plot
* Model performance metrics

## 🚀 Running the Project

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will open in the browser and display the dataset, model performance, predictions, and visualizations.

## 📈 Outcome

The project demonstrates how historical order data can be processed and used with a regression model to predict order demand, while Streamlit provides a simple browser-based interface for presenting the results.
