import time
import math
import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from tensorflow import keras
from datetime import date
from plotly import graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM



START = "2015-01-01"
TODAY = dt.datetime.now().strftime("%Y-%m-%d")

stocks = ["RELIANCE.NS"]


# Loading Data ---------------------

#@st.cache(suppress_st_warning=True)
def load_data(ticker):
    data = yf.download(ticker, START,  TODAY)
    data.reset_index(inplace=True)
    return data


#For Stock Financials ----------------------

def stock_financials(stock):
    df_ticker = yf.Ticker(stock)
    sector = df_ticker.info['sector']
    prevClose = df_ticker.info['previousClose']
    marketCap = df_ticker.info['marketCap']
    twoHunDayAvg = df_ticker.info['twoHundredDayAverage']
    fiftyTwoWeekHigh = df_ticker.info['fiftyTwoWeekHigh']
    fiftyTwoWeekLow = df_ticker.info['fiftyTwoWeekLow']
    Name = df_ticker.info['longName']
    averageVolume = df_ticker.info['averageVolume']
    ftWeekChange = df_ticker.info['52WeekChange']
    website = df_ticker.info['website']

    st.write('Company Name -', Name)
    st.write('Sector -', sector)
    st.write('Company Website -', website)
    st.write('Average Volume -', averageVolume)
    st.write('Market Cap -', marketCap)
    st.write('Previous Close -', prevClose)
    st.write('52 Week Change -', ftWeekChange)
    st.write('52 Week High -', fiftyTwoWeekHigh)
    st.write('52 Week Low -', fiftyTwoWeekLow)
    st.write('200 Day Average -', twoHunDayAvg)


#Plotting Raw Data ---------------------------------------

def plot_raw_data(stock, data_1):
    df_ticker = yf.Ticker(stock)
    Name = df_ticker.info['longName']
    #data1 = df_ticker.history()
    data_1.reset_index()
    #st.write(data_1)
    numeric_df = data_1.select_dtypes(['float', 'int'])
    numeric_cols = numeric_df.columns.tolist()
    st.markdown('')
    st.markdown('**_Features_** you want to **_Plot_**')
    features_selected = st.multiselect("", numeric_cols)
    if st.button("Generate Plot"):
        cust_data = data_1[features_selected]
        plotly_figure = px.line(data_frame=cust_data, x=data_1['Date'], y=features_selected,
                                title= Name + ' ' + '<i>timeline</i>')
        plotly_figure.update_layout(title = {'y':0.9,'x':0.5, 'xanchor': 'center', 'yanchor': 'top'})
        plotly_figure.update_xaxes(title_text='Date')
        plotly_figure.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, title="Price"), width=800, height=550)
        st.plotly_chart(plotly_figure)


#For LSTM MOdel ------------------------------

def create_train_test_LSTM(df, epoch, b_s, ticker_name):

    df_filtered = df.filter(['Close'])
    dataset = df_filtered.values

    #Training Data
    training_data_len = math.ceil(len(dataset) * .7)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0: training_data_len, :]

    x_train_data, y_train_data = [], []

    for i in range(60, len(train_data)):
        x_train_data.append(train_data[i-60:i, 0])
        y_train_data.append(train_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

    x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

    #Testing Data
    test_data = scaled_data[training_data_len - 60:, :]

    x_test_data = []
    y_test_data = dataset[training_data_len:, :]
    
    for j in range(60, len(test_data)):
        x_test_data.append(test_data[j - 60:j, 0])

    x_test_data = np.array(x_test_data)

    x_test_data = np.reshape(x_test_data, (x_test_data.shape[0], x_test_data.shape[1], 1))


    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))

    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train_data, y_train_data, batch_size=int(b_s), epochs=int(epoch))
    st.success("Your Model is Trained Succesfully!")
    st.markdown('')
    st.write("Predicted vs Actual Results for LSTM")
    st.write("Stock Prediction for 30 days - ",ticker_name)

    predictions = model.predict(x_test_data)
    predictions = scaler.inverse_transform(predictions)

    train = df_filtered[:training_data_len]
    valid = df_filtered[training_data_len:]
    valid['Predictions'] = predictions

    new_valid = valid[::-1]
    rev_valid = new_valid[:30]
    rev_valid_30 = rev_valid[::-1]
    n_valid = rev_valid_30.reset_index()
    n_valid.drop('index', inplace=True, axis=1)
    st.dataframe(n_valid)
    st.markdown('')
    st.write("Plotting Actual vs Predicted ")

    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize=(14, 8))
    plt.title('Actual Close prices vs Predicted Using LSTM Model', fontsize=20)
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Actual', 'Predictions'], loc='upper left', prop={"size":20})
    st.pyplot()



#Creating Training and Testing Data for other Models ----------------------

def create_train_test_data(df1):

    data = df1.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df1)), columns=['Date', 'High', 'Low', 'Open', 'Volume', 'Close'])

    for i in range(0, len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['High'][i] = data['High'][i]
        new_data['Low'][i] = data['Low'][i]
        new_data['Open'][i] = data['Open'][i]
        new_data['Volume'][i] = data['Volume'][i]
        new_data['Close'][i] = data['Close'][i]

    #Removing the hour, minute and second
    new_data['Date'] = pd.to_datetime(new_data['Date']).dt.date

    train_data_len = math.ceil(len(new_data) * .8)

    train_data = new_data[:train_data_len]
    test_data = new_data[train_data_len:]
    return train_data, test_data

#Plotting the Predictions -------------------------


def prediction_plot(pred_data, test_data, models, ticker_name):

    test_data['Predicted'] = 0
    test_data['Predicted'] = pred_data

    #Resetting the index
    test_data.reset_index(inplace=True, drop=True)
    st.success("Your Model is Trained Succesfully!")
    st.markdown('')
    st.write("Predicted Price vs Actual Close Price Results for - " ,models)
    st.write("Stock Prediction for next 30 days - ", ticker_name)
    st.write(test_data[['Date', 'Close', 'Predicted']])
    st.write("Plotting Close Price vs Predicted Price for - ", models)

    #Plotting the Graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Predicted'], mode='lines', name='Predicted'))
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), height=550, width=800,
                      autosize=False, margin=dict(l=25, r=75, b=100, t=0))

    st.plotly_chart(fig)



# Sidebar Menu -----------------------

menu=["EDA","Train Model"]
st.sidebar.title("Settings")
st.sidebar.subheader("Timeseries Settings")
choices = st.sidebar.selectbox("Select the Activity", menu,index=0)



if choices == 'EDA':
    
    # Importing dataset
    reliance_0 = yf.download('RELIANCE.NS', start='2015-01-01')
    reliance_0.reset_index(inplace = True)
    st.title('Reliance Stock Market Prediction')
    st.header("Data We collected from the source")
    st.write(reliance_0)

    reliance_1=reliance_0.drop(["Adj Close"],axis=1).reset_index(drop=True)
    reliance_2=reliance_1.dropna().reset_index(drop=True)

    reliance=reliance_2.copy()
    reliance['Date']=pd.to_datetime(reliance['Date'],format='%Y-%m-%d')
    reliance=reliance.set_index('Date')
    st.title('EDA')
    st.write(reliance)


# ---------------------------Graphs--------------------------------------

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Visualizations')

    st.header("Graphs")
    plt.figure(figsize=(20,10))
    #Plot 1
    plt.subplot(2,2,1)
    plt.plot(reliance['Open'],color='green')
    plt.xlabel('Date')
    plt.ylabel('Open Price')
    plt.title('Open')
    #Plot 2
    plt.subplot(2,2,2)
    plt.plot(reliance['Close'],color='red')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Close')
    #Plot 3
    plt.subplot(2,2,3)
    plt.plot(reliance['High'],color='green')
    plt.xlabel('Date')
    plt.ylabel('High Price')
    plt.title('High')
    #Plot 4
    plt.subplot(2,2,4)
    plt.plot(reliance['Low'],color='red')
    plt.xlabel('Date')
    plt.ylabel('Low Price')
    plt.title('Low')
    st.pyplot()

# ------------------------box-plots---------------------------------

    # Creating box-plots
    st.header("Box Plots")

    plt.figure(figsize=(20,10))
    #Plot 1
    plt.subplot(2,2,1)
    plt.boxplot(reliance['Open'])
    plt.xlabel('Date')
    plt.ylabel('Open Price')
    plt.title('Open')
    #Plot 2
    plt.subplot(2,2,2)
    plt.boxplot(reliance['Close'])
    plt.xlabel('Date')
    plt.ylabel('Cloes Price')
    plt.title('Close')
    #Plot 3
    plt.subplot(2,2,3)
    plt.boxplot(reliance['High'])
    plt.xlabel('Date')
    plt.ylabel('High Price')
    plt.title('High')
    #Plot 4
    plt.subplot(2,2,4)
    plt.boxplot(reliance['Low'])
    plt.xlabel('Date')
    plt.ylabel('Low Price')
    plt.title('Low')
    st.pyplot()

# ----------------------Histogram---------------------------------------

    st.header("Histogram")
    # Ploting Histogram
    plt.figure(figsize=(20,10))
    #Plot 1
    plt.subplot(2,2,1)
    plt.hist(reliance['Open'],bins=50, color='green')
    plt.xlabel("Open Price")
    plt.ylabel("Frequency")
    plt.title('Open')
    #Plot 2
    plt.subplot(2,2,2)
    plt.hist(reliance['Close'],bins=50, color='red')
    plt.xlabel("Close Price")
    plt.ylabel("Frequency")
    plt.title('Close')
    #Plot 3
    plt.subplot(2,2,3)
    plt.hist(reliance['High'],bins=50, color='green')
    plt.xlabel("High Price")
    plt.ylabel("Frequency")
    plt.title('High')
    #Plot 4
    plt.subplot(2,2,4)
    plt.hist(reliance['Low'],bins=50, color='red')
    plt.xlabel("Low Price")
    plt.ylabel("Frequency")
    plt.title('Low')
    st.pyplot()


# -------------------------KDE Plots-----------------------------------------

    st.header("KDE Plots")
    # KDE-Plots
    plt.figure(figsize=(20,10))
    #Plot 1
    plt.subplot(2,2,1)
    sns.kdeplot(reliance['Open'], color='green')
    plt.title('Open')
    #Plot 2
    plt.subplot(2,2,2)
    sns.kdeplot(reliance['Close'], color='red')
    plt.title('Close')
    #Plot 3
    plt.subplot(2,2,3)
    sns.kdeplot(reliance['High'], color='green')
    plt.title('High')
    #Plot 4
    plt.subplot(2,2,4)
    sns.kdeplot(reliance['Low'], color='red')
    plt.title('Low')
    st.pyplot()


    st.header('Years vs Volume')
    st.line_chart(reliance['Volume'])


# -------------------Finding long-term and short-term trends---------------------

    st.title('Finding long-term and short-term trends')
    reliance_ma=reliance.copy()
    reliance_ma['30-day MA']=reliance['Close'].rolling(window=30).mean()
    reliance_ma['200-day MA']=reliance['Close'].rolling(window=200).mean()

    st.write(reliance_ma)


    st.subheader('Stock Price vs 30-day Moving Average')
    plt.plot(reliance_ma['Close'],label='Original data')
    plt.plot(reliance_ma['30-day MA'],label='30-MA')
    plt.legend()
    plt.title('Stock Price vs 30-day Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot()


    st.subheader('Stock Price vs 200-day Moving Average')
    plt.plot(reliance_ma['Close'],label='Original data')
    plt.plot(reliance_ma['200-day MA'],label='200-MA')
    plt.legend()
    plt.title('Stock Price vs 200-day Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot()
# -----------------------------Train Model----------------------------------

elif choices == 'Train Model':
    st.subheader("Train Machine Learning Models for Stock Prediction")
    st.markdown('')
    st.markdown('**_Select_ _Stocks_ _to_ Train**')
    stock_select = st.selectbox("", stocks, index=0)
    df1 = load_data(stock_select)
    df1 = df1.reset_index()
    df1['Date'] = pd.to_datetime(df1['Date']).dt.date
    options = ['LSTM']
    st.markdown('')
    st.markdown('**_Select_ _Machine_ _Learning_ _Algorithms_ to Train**')
    models = st.selectbox("", options)
    submit = st.button('Train Model')

    if models == 'LSTM':
        st.markdown('')
        st.markdown('')
        st.markdown("**Select the _Number_ _of_ _epochs_ and _batch_ _size_ for _training_ form the following**")
        st.markdown('')
        epoch = st.slider("Epochs", 0, 300, step=1)
        b_s = st.slider("Batch Size", 32, 1024, step=1)
        if submit:
            st.write('**Your _final_ _dataframe_ _for_ Training**')
            st.write(df1[['Date','Close']])
            create_train_test_LSTM(df1, epoch, b_s, stock_select)

# -------------------------------------------------------------------------------------------
