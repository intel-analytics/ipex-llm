## Summary
This is the example to demonstrates how to use Analytics Zoo Orca data preprocesing APIs.
For the detail guide of Orca data preprocessing, please refer to [here](https://analytics-zoo.github.io/master/#Orca/data).

## Install Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__.

## Data Preparation
We use one of the datasets in Numenta Anomaly Benchmark ([NAB](https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv)) for demo. The NYC taxi passengers dataset contains 10320 records, each indicating the total number of taxi passengers in NYC at specific time.
Before you run the example, download the data(nyc_taxi.csv), and put into a directory.

## Run after pip install
You can easily use the following commands to run this example:
```bash
nyc_path=the file path to nyc_taxi.csv
python spark_pandas.py -f ${nyc_path} 
```
