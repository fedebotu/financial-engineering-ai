# 💸 Financial Engineering AI Projects

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/>
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" />
</p>

This repository collects projects for the course IE471 "Introduction to Financial Engineering" at KAIST.

## 🗂 Contents
Here is the list of projects:
1. Stock price prediction
2. Stock price prediction with sentiment analysis

## 🔮 Stock Price Prediction
We want to predict Samsung's adjusted closing stock price for 2019-2020 with data observations from 2000 to 2018. We experiment with LSTM and GRU models.
We also predict the latest Microsoft and Tesla stock price: the latter is particularly difficult given the recent surge in popularity of Elon Musk's company.

<p align="center">
  <img src="https://github.com/Juju-botu/financial-engineering-ai/blob/master/stock_price_prediction/images/samsung_all.jpg" alt="Samsung stocks all period" width=600px>
</p>

## 😃 Stock Price Prediction with Sentiment Analysis
We want to predict stock prices with the help of _sentiment analysis_ based on Twitter data. In particular, we predict Tesla, Inc. adjusted closing stock price in the last quarter of 2020  using data from the first three quarters of 2020. We also repeat the experiment with Nvidia and experiment with different recurrent architectures: RNN, GRU and LSTM.

<p align="center">
  <img src="https://github.com/Juju-botu/financial-engineering-ai/blob/master/stock_price_prediction_with_sentiment_analysis/images/wordcloud_tesla.png" alt="Word cloud of Tesla-related words on Twitter" width=600px>
</p>

## 📊 Clustering Bank Customers and Predicting their Loan Status
The goal of this project is to cluster bank customers to obtain information about the major correlations about their information, such as previous credit history, job, age and so on. Then, we predict their loan status by using Boosting algorithms such as XGBoost and show the results with XAI (Explainable AI).

<p align="center">
  <img src="https://github.com/Juju-botu/financial-engineering-ai/blob/master/clustering_prediction_loans/images/summary_plot_kaggle-1.png" alt="Summary Plot with XAI" width=600px>
</p>

## 📈  Market Analysis and Portfolio Optimization
In this project, we use deep learning to predict the future behavior of various stocks in the market. The idea is to optimize our portfolio based on these predictions: we show that by doing so we can outperform the equally weighted and capitalization weighted portfolios.

<p align="center">
  <img src="https://github.com/Juju-botu/financial-engineering-ai/blob/master/portfolio_optimization/images/portfolio_multi_new.png" alt="Porfolio Optimization" width=700px>
</p>

All the projects contain a report and its source code as well. If you would like to contribute to make them better, feel free to raise an `Issue` or to contribute with a `Pull Request`!
