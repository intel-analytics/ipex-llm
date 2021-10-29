## Stock prediction  use case in Chronos

---
We demonstrate how to use Chronos to predict stock prices based on historical time series data.

In the reference use case, we use the publicly available daily stock price of S&P500 stocks during 2013-2018 ([data source](https://www.kaggle.com/camnugent/sandp500/))

This use case example contains 2 notebook:

- **stock_prediction.ipynb** demonstrates how to leverage Chronos's built-in models ie. LSTM, to do time series forecasting. Reference: https://github.com/jwkanggist/tf-keras-stock-pred
- **stock_prediction_prophet.ipynb** demonstrates how to leverage Chronos's built-in models ie. Prophet, to do time series forecasting. Reference: https://github.com/facebook/prophet, https://github.com/jwkanggist/tf-keras-stock-pred

### Install

You can refer to Chronos installation document [here](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/chronos.html#install).

### Prepare dataset
- run `get_data.sh` to download the full dataset. It will download daily stock price of S&P500 stocks during 2013-2018 into data folder, preprocess and merge them into a single csv file `all_stocks_5yr.csv`.

