[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rajveersinghcse-reliance-stock-market-prediction.streamlit.app/)
[![MIT LICENSE](https://badgen.net//badge/license/MIT/green)](https://github.com/rajveersinghcse/Reliance_Stock_Market_Prediction/blob/main/LICENSE)   ![MAINTAINED BADGE](https://img.shields.io/badge/Maintained%3F-yes-green.svg)

<img height="25" width="80" src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"> <img height="25" width="70" src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white"> <img height="25" width="80" src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black"> <img height="25" width="70" src="https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white"> <img height="25" width="110" src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white"> <img height="25" width="100" src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white"> <img height="25" width="70" src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white"> <img height="25" width="90" src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white"> 

# Reliance-Stock-Market-Prediction 

![Banner](https://github.com/rajveersinghcse/rajveersinghcse/blob/master/img/StockMarker.jpg)

<h3>Hey Folks,👨🏻‍💻</h3>
<p>The project "Reliance Stock Market Prediction" is a notable achievement that focuses on predicting the stock prices of Reliance Industries Limited for the next 30 days. I developed this project during an internship, utilizing a variety of data analysis and machine learning techniques to forecast stock prices based on historical data.</p>

# Description of The Project:

<h3><b>Business Objective of the project</b></h3>
The main objective of the project is to predict the stock prices of Reliance Industries Limited for the upcoming 30 days. This involves obtaining essential stock data such as Open, High, Low, and Close prices from 2015 to 2022. The project aims to identify short-term and long-term trends, analyze the impact of external factors and significant events, and provide a forecast for the next month.

# Description of The Data:

- The project utilizes stock market <b>[DATA](https://finance.yahoo.com/quote/RELIANCE.NS/history?period1=1420070400&period2=1672444800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true)</b> gathered from January 1, 2015, to February 28, 2023.
- You can download the data from the above website or use the Yfinance library to collect the data.

## About Data 📈 

### This data includes attributes such as:
- <b>Date:</b> Date of trade
- <b>Open:</b> Opening price of the stock
- <b>High:</b> Highest price of the stock on that day
- <b>Low:</b> Lowest price of the stock on that day
- <b>Close:</b> Close price adjusted for splits
- <b>Adj Close:</b> Adjusted close price adjusted for splits, dividends, and capital gain distributions
- <b>Volume:</b> Volume of stock traded on that day


# Tools and Technologies: 

#### The project employs various tools and technologies, including:
- Python programming language
- Libraries such as NumPy, Matplotlib, SciPy, scikit-learn, TensorFlow, Keras, and Streamlit
- SVR, Random Forest, KNN, LSTM, and GRU models for stock price prediction
- Deployment using Streamlit

## How to install these libraries?
### You can install these libraries by using the command.

- You can install all the libraries in your system which I have used in my project by using only one command. 
- You will need Python in your system to use this command. You can use this given link to install Python in your system : [Python](https://www.python.org/downloads/)
- After installation of Python, you need to run this command in your command prompt.

```bash
pip install -r requirements.txt 
```
# Model Building:

- The project includes the development and evaluation of multiple machine learning models, including SVR, Random Forest, KNN, LSTM, and GRU.
- The LSTM model demonstrated higher accuracy compared to other models and was chosen for deployment.
<img height="170" width="350" src="https://github.com/rajveersinghcse/rajveersinghcse/blob/master/img/ModelBuilding.png" alt="ModelBuilding">

# Deployment:

- The project is deployed using Streamlit, allowing users to interact with the model and make stock price predictions based on historical data.
- The deployment version of the project is accessible through a provided link :[Project](https://rajveersinghcse-reliance-stock-market-prediction.streamlit.app/)

# Running the Project:

- To run the project locally, users can install the required libraries using the provided command in the command prompt. Use the below Streamlit command to launch the application.
```bash
streamlit run app.py 
```
### This project showcases a fusion of data analysis, machine learning, and software development skills to address the complex task of stock price prediction. It highlights the potential for using historical data and advanced models to make informed predictions about financial markets.
---
<p align="center">
<b>Enjoy Coding</b>❤
</p>
