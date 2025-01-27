import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go

# Title and Description
st.title("Sales Forecasting App")
st.write("This application forecasts sales using the Prophet model.")

# Upload CSV Files
uploaded_file = st.file_uploader("Upload your sales data (CSV format):", type=["csv"])

if uploaded_file:
    # Load the data
    data = pd.read_csv(uploaded_file)
    
    # Convert 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    # Show the uploaded data
    st.write("Uploaded Data:")
    st.write(data.head())

    # Prepare data for Prophet
    sales_data = data[['Sales']].reset_index()
    sales_data.columns = ['ds', 'y']  # Prophet requires 'ds' and 'y' column names

    # Train the model
    st.write("Training the Prophet model...")
    model = Prophet()
    model.fit(sales_data)

    # Forecast
    future = model.make_future_dataframe(periods=30)  # Forecast for the next 30 days
    forecast = model.predict(future)

    # Show forecasted data
    st.write("Forecasted Data:")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # Plot the forecast
    st.write("Forecast Plot:")
    fig1 = plot_plotly(model, forecast)  # Interactive plot using Plotly
    st.plotly_chart(fig1)

    # Plot components
    st.write("Forecast Components:")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)
else:
    st.warning("Please upload a CSV file to proceed.")
  