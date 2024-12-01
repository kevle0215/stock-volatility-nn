from datetime import datetime, timedelta
from urllib.error import URLError
import streamlit as st
import pandas as pd
import yfinance as yf
import altair as alt
from keras import models
import numpy as np


st.title("Volatility Analysis")

def get_data(symbol):
    try:
        # Extract data from yfinance
        df = yf.download(symbol)

        # Remove "Date" as a column
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Date'}, inplace=True)
        df = df.iloc[:, 2:]

        # Rename columns
        df.columns = ["close", "high", "low", "open", "volume"]

        # Cleanup data/add columns
        df['return'] = df['close'].pct_change().fillna(0)
        df['volatility'] = df['return'].rolling(window=30).std()
        df = df.dropna(subset=['volatility']).reset_index(drop=True)

        return df
    except Exception as e:
        st.error("Enter a Valid Stock")

def process_data(df):
    if len(df) < 100:
        raise ValueError("Not enough data to process. Need at least 100 rows.")
    
    X = []
    window_size = 100
    for i in range(1, len(df) - window_size - 1, 1):
        firstHigh = df.iloc[0, 1] 
        firstVolume = df.iloc[0, 4] if df.iloc[0, 4] > 0 else 1 

        temp = []
        for j in range(window_size):
            temp.append([
                (df.iloc[i + j, 1] - firstHigh) / firstHigh,
                (df.iloc[i + j, 4] - firstVolume) / firstVolume,
            ])
        X.append(np.array(temp).reshape(100, 2))
        
    X = np.array(X)
    return X.reshape(X.shape[0], 1, 100, 2)
        
def create_model():
    model = models.load_model('pages/volatility_model.keras')
    
    return model

try:
    stock = st.text_input("Enter a Stock to Analyze")

    if not stock:
        st.error("Please Enter a Stock!")
    else: 
        df = get_data(stock)
        if df is not None:
            model = create_model()
            X = process_data(df)
            predicted_volatility_raw = model.predict(X)

            # Flatten the predictions
            predicted_volatility = predicted_volatility_raw.flatten()

            # Display predicted volatility for the current 100-day period
            st.write(f"Predicted Volatility For Current 100-Day Period: {predicted_volatility[len(predicted_volatility)-1]}")

            # Convert predictions to a df for Altair
            predicted_volatility_df = pd.DataFrame({
                'volatility': predicted_volatility
            })
            
            # Stock volatility descriptions
            st.table(predicted_volatility_df.describe())

            # Historical volatility chart 
            bar_chart = alt.Chart(predicted_volatility_df).mark_bar().encode(
                x=alt.X('volatility:Q', bin=alt.Bin(step=0.10), title='Volatility'),
                y=alt.Y('count():Q', title='Count')
            ).properties(
                title="Predicted Volatility Distribution",
                width=700,
                height=400
            )

            # Render the bar chart in Streamlit
            st.altair_chart(bar_chart, use_container_width=True)

        else:
            st.error("Unable to fetch data. Check the stock symbol.")
except URLError as e:
    st.error(f"This demo requires internet access. Connection error: {e.reason}")
except Exception as e:
    st.error(f"An error occurred while processing the data. {e}")