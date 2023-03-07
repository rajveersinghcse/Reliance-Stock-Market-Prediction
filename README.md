# Reliance-Stock-Market-Prediction
💹Predication of Reliance Stock Price Project.


<h3>Hey Folks,👨🏻‍💻</h3>
<p>I have created a <b>Reliance Stock Market Price Prediction</b> project that can predict the stock price of any company for the next 30 days. Here I used the last 7 years' data of Reliance Industries Limited. I did this project during my internship</p>

<h2>Description of the project:</h2>
<h4>Business Objective of the project</h4>

- Predict the Reliance Industries Stock Price for the next 30 days.

- There are Open, High, Low and Close prices that you need to obtain from the web for each day starting from 2015 to 2022 for Reliance Industries stock.

- Split the last year into a test set- to build a model to predict stock price.

- Find short term, & long term trends.

- Understand how it is impacted from external factors or any big external events.

- Forecast for next 30 days.

# About Data 📈 

- [Data.](https://finance.yahoo.com/quote/RELIANCE.NS/history?period1=1420070400&period2=1672444800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true) I collected this data from Jan-2015 to Current date.

- You can download the data from the above website or use the Yfinance library to collect the data.

- Date: Date of trade

- Open: Opening Price of Stock

- High: Highest price of stock on that day

- Low: Lowest price of stock on that day

- Close: Close price adjusted for splits.

- Adj Close: Adjusted close price adjusted for splits and dividend and/or capital gain distributions.

- Volume: Volume of stock on that day


# Libraries that i used in the project. 
- I used diffrent libraries in this project

##### You can install these libraries by using the command.
- It can install all the libraries in your system which I have used in my project. You will need Python in your system to use this command. You can use this given link to install python in your system : [Python](https://www.python.org/downloads/)
```bash
pip install -r requirements.txt 
```
# How to deploye the project.
- We used Streamlit Library to deployemnt part of this project. To deploye or run this project in your local system you have to run this command into your command prompt.
```bash
streamlit run app.py 
```

---
<p align="center">
<b>Enjoy Coding</b>❤
</p>
