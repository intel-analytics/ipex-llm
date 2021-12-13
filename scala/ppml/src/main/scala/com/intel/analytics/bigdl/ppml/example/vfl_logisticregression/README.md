# VFL Logistic Regression Example

This example show how to create an end-to-end VFL Logistic Regression application with 2 clients on BigDL PPML.

## Data
### Prepare dataset

First, go to the working directory to prepare data, to show a 2-clients workflow, we split the dataset into 2 parts.

We use the [diabetes dataset from Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database). To ease you from the procedure of downloading from Kaggle, here we download the same data from [another website](http://networkrepository.com/pima-indians-diabetes.php):
```bash
wget http://nrvis.com/data/mldata/pima-indians-diabetes.csv 
```

Run the script ```split_data.py``` to split features of the dataset, and create two corresponding smaller datasets, you may install pandas package necessary for running the script by ```pip3 install pandas```:
```bash
python3 split_dataset.py 
```
Then, you will be prompted to type the original data file name, if you press 'ENTER', the script will use the defualt value "pima-indians-diabetes.csv", you can also type names of other data files to be splitted.
By default, the script will put column index 0,1,2,3,8 of "pima-indians-diabetes.csv" in data file 1, and the rest of them in file 2, if you want to split on customed column indexes, you need to specify it in the command line, seperate them with spaces, for example:
```bash
python3 split_dataset.py 0 2 4 6
```

Then check the files in your current path, there should be two newly generated csv files called ```diabetes-1.csv``` and ```diabetes-2.csv```. Before feeding them into the model, we need to add key for each row. You can refer to [here](#binaddrowkeysh) for detailed explanation. Run commands below:
```
bash addRowKey.sh RowKey diabetes-1.csv
bash addRowKey.sh RowKey diabetes-2.csv
```
Finally, create a new directory `dataset/diabetes` to put the dataset:
```bash
mkdir ../dataset/diabetes
mv diabetes* ../dataset/diabetes
cd ..
```
The data loading method is provided in code.

## Start FLServer
Before starting server, modify the config file, `ppml-conf.yaml`, this application has 2 clients globally, so use following config.
```
worldSize: 2
```
Then start FLServer at server machine
```
java -cp bigdl-ppml-xxx.jar com.intel.analytics.bigdl.ppml.FLServer
```

## Start Local Trainers
Change the config file to following. 
```
clientTarget: FLServer_URL
```
The port of server is provided in `ppml-conf.yaml` when server starts, default value `8980`. e.g. if you run the server and clients on same machine, `FLServer_URL` should be `localhost:8980`

And start the local Logistic Regression trainers at 2 training machines, with learning rate 0.01, batch size 4
```
java -cp bigdl-ppml-xxx.jar com.intel.analytics.bigdl.ppml.example.logisticregression.VflLogisticRegression 
    --dataPath dataset/diabetes/diabetes-1.csv 
    --rowKeyName ID
    --learningRate 0.005
    --batchSize 4    
# change dataPath to diabetes-2.csv at client-2
```

The example will train the data and evaluate the training result.