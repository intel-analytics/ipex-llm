# Web Service Sample

## Summary
This is a web service sample for text classification model. 
Briefly speaking, after starting the web service application, user can post a request body contains an article texts to the server's url followed with directory "/predict",
(eg: localhost:8080/predict). 
Then the server application will do a series of actions including preprocessing the texts, loading the model and doing the prediction.
In the end, it will response with the predicted class and predicted probability distribution of the tested texts.

In this directory, there are two packages
1. analytics-zoo-inference-example is the web application sample project.
2. analytics-zoo-preprocess is the utility package for text preprocessing.

To run this sample, please follow the steps below. 

## Start up the web service application
### Import Project
In the IDE(eg:IDEA), new a project from existing source, and choose the pom.xml in analytics-zoo-inference-example, using maven to build up the project.

### Prepare Data 
Download the word vectorized embedding map, for example:
   - [GloVe word embeddings(glove.6B.zip)](http://nlp.stanford.edu/data/glove.6B.zip): embeddings of 400k words pre-trained on a 2014 dump of English Wikipedia.
   
   You need to prepare the data by yourself beforehand. The following scripts we prepare will serve to download and extract the data:
   ```
   bash ${ANALYTICS_ZOO_HOME}/bin/data/glove/get_glove.sh dir
   ```
   where `ANALYTICS_ZOO_HOME` is the `dist` directory under the Analytics Zoo project and `dir` is the directory you wish to locate the downloaded data. If `dir` is not specified, the data will be downloaded to the current working directory. 
   Set the environment variable as follow, this can be done either by editing the run/debug configurations or exporting into system:
   ```
   EMBEDDING_PATH="the file path of the embedding map"
   ```
   
### Prepare Model (Optional)
* This sample will use the pre-trained model in resources as default, therefore, preparing the model is not a must.

Otherwise, you can prepare the text classification model by yourself, see [Text Classification Model Trainning Example](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/textclassification).
And don't forget to set the model path as the environment variable as below:
```
modelPath="the file path of text classification model"
```
### Run Application
Run the Application.java, the web service application will be started. 
Then, from the browser end, when posted the texts as the request body, the application will respond with the prediction result.







