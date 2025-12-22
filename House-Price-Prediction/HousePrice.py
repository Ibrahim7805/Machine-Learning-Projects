import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score , mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

# Add title
st.title('House Price prediction Using LR...')

# Add Header
st.header('User Input')

uploaded_data = st.file_uploader("Please upload your data...", type='csv')

if uploaded_data is not None:
    data =pd.read_csv(uploaded_data)
    st.write('The 10 five rows of your data.. ', data.head(10))

    features = st.multiselect('Select your features', data.columns)
    Target = st.multiselect('Select your Target', data.columns)

    X = data[features]
    Y = data[Target]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.2, random_state=42)

# ______________________________________________

    model = LinearRegression()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write('MAE: ', mean_absolute_error(y_test, y_pred))
    st.write('R2_Score: ', r2_score(y_test, y_pred))

# ______________________________________________

    fig, ax = plt.subplots()
    ax.scatter(x=y_pred, y=y_test)
    ax.set_xlabel("prediction")
    ax.set_xlabel("Target")

    st.pyplot(fig)



