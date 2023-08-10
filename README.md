[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rajveersinghcse-reliance-stock-market-prediction.streamlit.app/)
[![MIT LICENSE](https://badgen.net//badge/license/MIT/green)](https://github.com/rajveersinghcse/Reliance_Stock_Market_Prediction/blob/main/LICENSE)   ![MAINTAINED BADGE](https://img.shields.io/badge/Maintained%3F-yes-green.svg) 

# Reliance-Stock-Market-Prediction 

![Banner](https://github.com/rajveersinghcse/rajveersinghcse/blob/master/img/StockMarker.jpg)

<h3>Hey Folks,👨🏻‍💻</h3>
<p>The project "Reliance Stock Market Prediction" is a notable achievement that focuses on predicting the stock prices of Reliance Industries Limited for the next 30 days. I developed this project during an internship, utilizing a variety of data analysis and machine learning techniques to forecast stock prices based on historical data.</p>

# Description of The Project:
<h3><b>Business Objective of the project</b></h3>
The main objective of the project is to predict the stock prices of Reliance Industries Limited for the upcoming 30 days. This involves obtaining essential stock data such as Open, High, Low, and Close prices from 2015 to 2022. The project aims to identify short-term and long-term trends, analyze the impact of external factors and significant events, and provide a forecast for the next month.

# Description of The Data?

- The project utilizes stock market <b>[DATA](https://finance.yahoo.com/quote/RELIANCE.NS/history?period1=1420070400&period2=1672444800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true)</b> gathered from January 1, 2015, to February 28, 2023.
- This data includes attributes such as:
- You can download the data from the above website or use the Yfinance library to collect the data.

## About Data 📈 

- <b>Date:</b> Date of trade
- <b>Open:</b> Opening price of the stock
- <b>High:</b> Highest price of the stock on that day
- <b>Low:</b> Lowest price of the stock on that day
- <b>Close:</b> Close price adjusted for splits
- <b>Adj Close:</b> Adjusted close price adjusted for splits, dividends, and capital gain distributions
- <b>Volume:</b> Volume of stock traded on that day


# Libraries that I used in the project. 
<img height="25" width="80" src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"> <img height="25" width="70" src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white"> <img height="25" width="80" src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black"> <img height="25" width="70" src="https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white"> <img height="25" width="110" src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white"> <img height="25" width="100" src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white"> <img height="25" width="70" src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white"> <img height="25" width="90" src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white"> 


## How to install these libraries?

### You can install these libraries by using the command.

- It can install all the libraries in your system which I have used in my project. 

- You will need Python in your system to use this command. You can use this given link to install Python in your system : [Python](https://www.python.org/downloads/)

- After installation of Python, you need to run this command in your command prompt.

```bash
pip install -r requirements.txt 
```
# Model Building.
- For the model building part, we used SVR, Random Forest, KNN, LSTM, and GRU models.

- I was getting more accuracy in LSTM than in other models. So I decided to use the LSTM model in my deployment program or main project.
<img height="170" width="350" src="https://github.com/rajveersinghcse/rajveersinghcse/blob/master/img/ModelBuilding.png" alt="ModelBuilding">

# Cloud version of this project.
- I deploy this project on the cloud you can check it out at this link: [Project](https://rajveersinghcse-reliance-stock-market-prediction.streamlit.app/)


# How to deploy the project?
- We used the Streamlit library for the deployment part of this project. To deploy or run this project in your local system, you must run this command in your command prompt.
```bash
streamlit run app.py 
```
---
<p align="center">
<b>Enjoy Coding</b>❤
</p>
