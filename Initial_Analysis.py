from urllib.error import URLError
import streamlit as st
import pandas as pd
import yfinance as yf
import altair as alt

st.title("Initial Analysis")

def get_data(symbol):
    try:
        # Extract data from yfinance
        df = yf.download(symbol)

        # Make "Date" a column, not an index
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Date'}, inplace=True)
        df = df.iloc[:, 0:]

        # Rename columns appropriately
        df.columns = ["Date", "alt close", "close", "Price", "low", "open", "volume"]

        # Cleanup data/add columns
        df['return'] = df['close'].pct_change().fillna(0)
        df['volatility'] = df['return'].rolling(window=30).std()
        df = df.dropna(subset=['volatility']).reset_index(drop=True)

        return df
    except Exception as e:
        st.error("Enter a Valid Stock")

try:
    stock = st.text_input("Enter a Stock to Analyze")
    
    if not stock:
        st.error("Please Enter a Stock!")
        
    else: 
        # Fetch data
        df = get_data(stock)
        
        # Turn data into dataframe
        st.dataframe(df)
        st.table(df.describe())

        # Price vs. Date line chart
        chart = alt.Chart(df).mark_line().encode(
            x='Date',
            y='Price'
        )
        st.altair_chart(chart, use_container_width=True)
        
        # Create Altair bar chart with numerical buckets
        bar_chart = alt.Chart(df[['return']]).mark_bar().encode(
            x=alt.X('return:Q', bin=alt.Bin(step=0.01), title='Return'),  # Binning
            y=alt.Y('count():Q', title='Count')  # Count of values in each bucket
        ).properties(
            title="Average Return",
            width=700,
            height=400
        )
        st.altair_chart(bar_chart, use_container_width=True)
        
except URLError as e:
    st.error(f"This demo requires internet access. Connection error: {e.reason}")
    
except Exception as e:
    st.error("Please Enter a Valid Stock")