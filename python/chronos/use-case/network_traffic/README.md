## Network traffic use case in Chronos

---
We demonstrate how to use Chronos to forecast future network traffic indicators based on historical
time series data. 

In the reference use case, we use a public telco dataset, which is aggregated network traffic traces at the transit link of WIDE
to the upstream ISP ([dataset link](http://mawi.wide.ad.jp/~agurim/dataset/)). 
 

This use case example contains two notebooks:

- **network_traffic_autots_forecasting.ipynb** demonstrates how to use `AutoTS` to automatically
generate the best end-to-end time series analysis pipeline.

- **network_traffic_model_forecasting.ipynb** demonstrates how to leverage Chronos's built-in models 
ie. LSTM and MTNet, to do time series forecasting. Both univariate and multivariate analysis are
demonstrated in the example.



### Install

You can refer to Chronos installation document [here](https://analytics-zoo.github.io/master/#Chronos/tutorials/Autots/#step-0-prepare-environment).

### Prepare dataset
* run `get_data.sh` to download the full dataset. It will download the monthly aggregated traffic data in year 2018 and 2019 (i.e "201801.agr", "201912.agr") into data folder. The raw data contains aggregated network traffic (average MBPs and total bytes) as well as other metrics.
* run `extract_data.sh` to extract relevant traffic KPI's from raw data, i.e. AvgRate for average use rate, and total for total bytes. The script will extract the KPI's with timestamps into `data/data.csv`.


