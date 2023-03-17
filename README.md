# Reliance-Stock-Market-Prediction ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

Predication of Reliance Stock Price Project.
![Banner](https://github.com/rajveersinghcse/rajveersinghcse/blob/master/img/StockMarker.jpg)

<h3>Hey Folks,👨🏻‍💻</h3>
<p>I have created a <b>Stock Market Price Prediction</b> project that can predict the stock price of any company for the next 30 days. Here I used the last 7 years' data of Reliance Industries Limited. I did this project during my internship</p>

# What we have to do in this Project:
<h3><b>Business Objective of the project</b></h3>

- Predict the Reliance Industries Stock Price for the next 30 days.

- There are Open, High, Low and Close prices that you need to obtain from the web for each day starting from 2015 to 2022 for Reliance Industries stock.

- Split the last year into a test set- to build a model to predict stock price.

- Find short term, & long term trends.

- Understand how it is impacted from external factors or any big external events.

- Forecast for next 30 days.

# How to collect the Data?

- I collected this data from 1-Jan-2015 to 28-Feb-2023. <b>[DATA](https://finance.yahoo.com/quote/RELIANCE.NS/history?period1=1420070400&period2=1672444800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true)</b>

- You can download the data from the above website or use the Yfinance library to collect the data.

# About Data 📈 

- Date: Date of trade

- Open: Opening Price of Stock

- High: Highest price of stock on that day

- Low: Lowest price of stock on that day

- Close: Close price adjusted for splits.

- Adj Close: Adjusted close price adjusted for splits and dividend and/or capital gain distributions.

- Volume: Volume of stock on that day


# Libraries that i used in the project. 
- I have already mentioned in my [Libraries](https://github.com/rajveersinghcse/Reliance_Stock_Market_Prediction/blob/main/requirements.txt) folder which libraries I have used in my project.

## How to install these libraries?

### You can install these libraries by using the command.

- It can install all the libraries in your system which I have used in my project. 

- You will need Python in your system to use this command. You can use this given link to install python in your system : [Python](https://www.python.org/downloads/)

- After installation of python, you need to run this command in your command prompt.

```bash
pip install -r requirements.txt 
```
# Model Building.
- For model building part, we used SVR, Random Forest, KNN, LSTM, and GRU models.

- I was getting more accuracy in LSTM than in other models. So I decided to use the LSTM model in my deployment program or main project.
<img height="170" width="350" src="https://github.com/rajveersinghcse/rajveersinghcse/blob/master/img/ModelBuilding.png" alt="ModelBuilding">

# Cloud Version of this project.
- I deploy this project on the cloud you can check it out at this link: [Project](https://rajveersinghcse-reliance-stock-market-prediction-app-0aobyl.streamlit.app/)


# How to deploy the project?
- We used Streamlit library for the deployment part of this project. To deploy or run this project in your local system, you must run this command in your command prompt.
```bash
streamlit run app.py 
```

---
<p align="center">
<b>Enjoy Coding</b>❤
</p>
