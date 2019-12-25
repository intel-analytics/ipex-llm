# Model Inference Examples

## Summary
This is examples for model inference, including model inference in simple pojo java application, web service, batch and streaming with sprak and flink.

Briefly speaking, for web service example, after starting the web service application, user can post a request body that contains an article text to the server's url followed with directory "/predict", (eg: localhost:8080/predict).
Then the server application will do a series of actions including preprocessing the texts, loading the model and doing the prediction.
In the end, it will respond with the predicted class and predicted probability distribution of the tested texts.

In this directory, there are several projects

1. `text-classification-training` is the utility project(written in Scala). It includes procedures of preprocessing and training.
2. `text-classification-inference` is the web application sample project(written in Java). It loads the model and does the prediction.
3. `model-inference-flink` is the model inference in batch and streaming with flink.

To run this sample, please follow the steps below.

## Start up the applications
### Import Project
In the IDE(eg: IDEA), Select New - Project from Existing Source, look through the directory to find pom.xml file of the project(eg: text-classification-training) and click OK, then select open as project in the window pop out next, using maven to build up the project.

### Prepare Data
The data used in this example are:

* [20 Newsgroup dataset(news-18828.tar.gz)](http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz) which contains 20 categories and with 18828 texts in total.
* [GloVe word embeddings(glove.6B.zip)](http://nlp.stanford.edu/data/glove.6B.zip): embeddings of 400k words pre-trained on a 2014 dump of English Wikipedia.

You need to prepare the data by yourself beforehand. The following scripts we prepare will serve to download and extract the data:

    bash ${ANALYTICS_ZOO_HOME}/bin/data/news20/get_news20.sh dir
    bash ${ANALYTICS_ZOO_HOME}/bin/data/glove/get_glove.sh dir

where `ANALYTICS_ZOO_HOME` is the folder where you extract the downloaded package and `dir` is the directory you wish to locate the downloaded data. If `dir` is not specified, the data will be downloaded to the current working directory. 20 Newsgroup dataset and GloVe word embeddings are supposed to be placed under the same directory.

The data folder structure after extraction should look like the following:

    baseDir$ tree .
	    .
	    ├── 20news-18828
	    └── glove.6B

### Training the Model in text-classification-training
To prepare the model, you need to import the `text-classification-training` project. After the maven dependencies are successfully downloaded, set the environment variable as follow, this can be done by editing the run/debug configurations:

    VM options: -Xmx20g
    Program arguments: --batchSize xxx --trainDataDir "/YOURDIR/data/20news-18828"/ --embeddingFile "/YOURDIR/data/glove/glove.6B.100d.txt" --modelSaveDirPath "/YOURDIR/models/text-classification.bigdl"

Note that the batch size must be the multiple of the system core number. You can check the core number of your system with the command `lscpu` and change the batch size by adding `--batchSize xxx` in Program arguments.

After the training is done, the model will be saved into the directory you set. Go back to the root directory of `text-classification-training` and execute the `mvn clean install` command, which prepares the jar file for `text-classification-inference`.

### Run Simple Java or Web Service examples in text-classification-training 
In `text-classification-inference`, there are two ways to run the application: `SimpleDriver.java` and `WebServiceDriver.java`.

In the first place, you need to edit the run/debug configurations as below:

    VM options: -DEMBEDDING_FILE_PATH=/YOURDIR/data/glove/glove.6B.50d.txt -DMODEL_PATH=/YOURDIR/models/text-classification.bigdl

For `SimpleDriver.java`, simply running it can get the prediction result in the terminal.

For `WebServiceDriver.java`, running it will start the web service application. To see the output, you can use tools such as `Postman` to send a POST request whose body contains an article text to the server's url followed with directory "/predict", (eg: localhost:8080/predict). Then the application will respond with the prediction result.

### Run flink examples in model-inference-flink
#### Run on Flink Cluster

1. you may start a flink cluster if there is no runing one:
    
    `./path/to/your/flink-1.7.2/bin/start-cluster.sh`

2. start a socket to pass words:

    `nc -l 9000`

3. build your flink job package and start the flink job for text classification model inference:

    ``` 
    mvn clean package
        
    ./path/to/your/flink-1.7.2/bin/flink run ./path/to/your/analytics-zoo/apps/model-inference-examples/model-inference-flink/target/model-inference-flink-0.1.0-SNAPSHOT-jar-with-dependencies.jar --port 9000 --embeddingFilePath ./path/to/your/data/glove/glove.6B.100d.txt --modelPath ./path/to/your/models/text-classification.bigdl --parallelism 2
    ```
4. tests with echo word in nc -l 9000 console and check the output with:

    `tail -f ./path/to/your/flink-1.7.2/log/flink-*-taskexecutor-*.out`

Or can use another method which don't need to use nc commend.

1. you may start a flink cluster if there is no runing one:
  
    `./path/to/your/flink-1.7.2/bin/start-cluster.sh`

2. build your flink job package and start the flink job for text classification model inference:

     ```
     mvn clean package  
     ./path/to/your/flink-1.7.2/bin/flink run \
     ./path/to/your/analytics-zoo/apps/model-inference-examples/model-inference-flink/target/model-inference-flink-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
     --embeddingFilePath ./path/to/your/data/glove/glove.6B.100d.txt \
     --modelPath ./path/to/your/models/text-classification.bigdl --parallelism 2 \
     --inputFile ./path/to/your/input-file
     ```

3. check the output with:

    `tail -f ./path/to/your/flink-1.7.2/log/flink-*-taskexecutor-*.out`
