# FGBoost Regression Quick Start
This example provides a step-by-step tutorial of running a FGBoost Regression task with 2 parties.
## Data
This section describe the data and preprocess, you may skip if you want to directly run the application.

This example use [House Price Dataset]().

To simulate the scenario where different data features are held by 2 parties respectively, we split the dataset to 2 parts and preprocess them in advance. The split is taken by select every other column (code at [split script]()) and preprocessing includes filling NA values and applying one-hot encoding to categorical features (code at [preprocess script]()).

After preprocessing, we got a data file in `data` folder, including `house-prices-train-1.csv` with half of features and no label, and a data file `house-prices-train-2.csv` with another half of features and label `SalePrice`. The and test data `house-prices-test-1.csv` and `house-prices-test-2.csv`.

## Start FLServer
FL Server is required before running any federated applications. Please check [Start FL Server]() section.

## Run Client Application
To simuluate the training of 2 parties, we provide 2 Python scripts which has the similar pipeline logic. The pipelines both include loading data, running federated training, running federated prediction, and save/load the model. Note that the codes of them are slightly different due to the different data file they access.

To run the example, start these 2 scripts from 2 different terminals.
```bash
python fgboost_regression_party_1.py
```
and 
```bash
python fgboost_regression_party_2.py
```
The order of starts does not matter.
## Code Step-by-step Client Code