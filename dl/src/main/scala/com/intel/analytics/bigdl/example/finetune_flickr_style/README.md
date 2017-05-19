# Fine-tuning pre-trained models for Style Recognition on "Flickr Style" Data

This example demonstrates how to use BigDL to finetune pre-trained [Caffe](http://caffe.berkeleyvision.org/) model on a different dataset, [Flickr Style](http://sergeykarayev.com/files/1311.3715v3.pdf), to predict image style instead of object category.

BigDL also support to finetune on other sorts of pre-trained models like Torch models and BigDL models.

This example refers to caffe example "fine tune flickr style ", but with BigDL implementation and modifications. 

The Flickr-sourced images of the Style dataset are visually very similar to the ImageNet dataset, on which the `bvlc caffenet` model was trained.
Since the caffenet model works well for image category classification, we'd like to use this architecture for the style classifier.
The Flickr-sourced images have 80,000 images to train on, so we'd like to start with the parameters learned on the ImageNet images, and fine-tune as needed.

## Prepare Data
The dataset is distributed as a list of URLs with corresponding labels.
Using a script, we will download a small subset of the data and split it into train and val sets.    

Before run such script, install python pandas and skimage on your system. 

i.e. run "sudo apt-get install python-pandas python-skimage" on ubuntu system.
    
 ```bash
Usage: gen_data.py [-h] [-d DEST] [-s SEED] [-i IMAGES] [-w WORKERS]
                              [-l LABELS]
Download a subset of Flickr Style to a directory

    optional arguments:
      -h, --help            show this help message and exit
      -d DEST, --dest DEST  destination directory
      -s SEED, --seed SEED  random seed
      -i IMAGES, --images IMAGES
                            number of images to use (-1 for all)
      -w WORKERS, --workers WORKERS
                            num workers used to download images. -x uses (all - x)
                            cores.

Example: python ./gen_data.py -d ~/data/flickr_style --workers=-1 --images=2000 --seed 831486
    Downloading 2000 images with 7 workers...
    Writing train/val for 1939 successfully downloaded images.
```    
Now all the images belong to the same category are moved to the destination folder.

If you want to do the distributed traing, you need to use below command to transform the images into hadoop sequence files, which are more suitable for a distributed environment.

```bash
java -cp bigdl_folder/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.models.finetune_flickr_style.ImageSeqFileGenerator -f image_folder -o output_folder -p cores_number -r
```

We'll also need to download the ImageNet-trained [CaffeNet](http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodelrunning) model. 

## Prepare the Model
Since we are finetuning based on pre-trained CaffeNet model, we need to create our BigDL model based on CaffeNet but do some modification.

Because we are predicting 20 classes instead of a 1,000, we do need to change the last layer in the model. Therefore, we change the name of the last layer from `fc8` to `fc8_flickr` in original CaffeNet model, and modify output size to 20.
Please look into the detail model definition in finetune_flickr_style/Models.scala.

We will also decrease the overall learning rate to have the the model change very slowly with new data.

Additionally, we set step size in the optimizer to a lower value than if we were training from scratch, since we're virtually far along in training and therefore want the learning rate to go down faster.

## Train the Model Locally
Use com.intel.analytics.bigdl.example.finetune_flickr_style.Train to train model.
```
Usage: BigDL FineTune Example [options]

  -f <value> | --folder <value>
        where you put your local data/seq files
  --model <value>
        model snapshot location
  --checkpoint <value>
        where to cache the model
  --state <value>
        state snapshot location
  -c <value> | --core <value>
        cores number to train the model
  --modelName <value>
        model name
  --caffeDefPath <value>
        caffe model definition file
  --modelPath <value>
        existing model path
  -b <value> | --batchSize <value>
        batch size
  -l <value> | --learningRate <value>
        Learning Rate
  -e <value> | --maxEpoch <value>
        epoch numbers
  --env <value>
        running environment, should be local or spark
  -n <value> | --node <value>
        node number to train the model
```        
### Train with pre-trained model
Now we use pre-trained CaffeNet to train our flickr style recognition. 
Example command :                     
```
dist/bin/bigdl.sh -- java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.example.finetune_flickr_style.Train -f ~/data/flickr_style --caffeDefPath ~/train_val.prototxt --modelPath ~/model/bvlc_reference_caffenet.caffemodel --core 4 --node 1 -b 52 --env local --checkpoint ~/model
```

Here is the result of training after 1 epoch:
```
2017-01-08 06:41:13 INFO  LocalOptimizer$:226 - [Validation] 13664/14091 Throughput is 315.36154392609774 record / sec
2017-01-08 06:41:22 INFO  LocalOptimizer$:226 - [Validation] 13888/14091 Throughput is 375.3895974233473 record / sec
2017-01-08 06:41:30 INFO  LocalOptimizer$:226 - [Validation] 14091/14091 Throughput is 276.7290160814088 record / sec
2017-01-08 06:41:30 INFO  LocalOptimizer$:235 - top1 accuracy is Accuracy(correct: 4405, count: 14091, accuracy: 0.3126108863813782)
2017-01-08 06:41:30 INFO  LocalOptimizer$:235 - top5 accuracy is Accuracy(correct: 9946, count: 14091, accuracy: 0.7058406074799517)
```
We can see only after training 1 epoch, the top1 accuracy has arrived at 0.3126, top5 accuracy reached 0.7058

### Train without pre-trained model
Now we can try to train the model from scratch. 
Example command:
```
dist/bin/bigdl.sh -- java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.example.finetune_flickr_style.Train -f ~/data/flickr_style --core 4 --node 1 -b 52--env local --checkpoint ~/model
```

Here is the result after 1 epoch:
```
2017-01-08 17:47:01 INFO  LocalOptimizer$:226 - [Validation] 13936/14091 Throughput is 326.92813427401205 record / sec
2017-01-08 17:47:07 INFO  LocalOptimizer$:226 - [Validation] 14091/14091 Throughput is 364.77574334501077 record / sec
2017-01-08 17:47:07 INFO  LocalOptimizer$:235 - top1 accuracy is Accuracy(correct: 867, count: 14091, accuracy: 0.0615286352991271)
2017-01-08 17:47:07 INFO  LocalOptimizer$:235 - top5 accuracy is Accuracy(correct: 4045, count: 14091, accuracy: 0.28706266411184445)
```
After training 1 epoches, we can see the top1 accuracy only arrived at 0.0615, top5 accuracy only reached 0.2870

## Train the Model Using Apache Spark
### Local Mode with Finetune
Example Command:
```
./dist/bin/bigdl.sh -- spark-submit --master local[4] --driver-class-path dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar --class com.intel.analytics.bigdl.example.finetune_flickr_style.Train dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar -f  ~/data/flickr_style --caffeDefPath ~/train_val.prototxt --modelPath ~/model/bvlc_reference_caffenet.caffemodel --core 4 --node 1 -b 52 --checkpoint ~/model --env spark
```
### Local Mode from Scratch
Example Command
```
./dist/bin/bigdl.sh -- spark-submit --master local[4] --driver-class-path dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar --class com.intel.analytics.bigdl.example.finetune_flickr_style.Train dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar -f  ~/data/flickr_style --core 4 --node 1 -b 52 --checkpoint ~/model --env spark
```
### Cluster Mode with Finetune
Example Command:
```
./dist/bin/bigdl.sh -- spark-submit --driver-class-path dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar --class com.intel.analytics.bigdl.example.finetune_flickr_style.Train dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar -f hdfs://data/flickr_style --caffeDefPath ~/train_val.prototxt --modelPath ~/model/bvlc_reference_caffenet.caffemodel --core 4 --node 4 --env spark -b 48
```

### Cluster Mode from Scratch
Example Command:
```
./dist/bin/bigdl.sh -- spark-submit --driver-class-path dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar --class com.intel.analytics.bigdl.example.finetune_flickr_style.Train dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar -f hdfs://data/flickr_style --core 4 --node 4 --env spark -b 48
```
## Test Model
The above commands will cache the model in specified path(--checkpoint). Run this command will
use the model to do a validation.

Example command
```
dist/bin/bigdl.sh -- java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.example.finetune_flickr_style.Test -f  ~/data/flickr_style --core 4 -n 1 --env local --model ~/model/model.5000
```
Spark local mode, example command
```
./dist/bin/bigdl.sh -- spark-submit --master local[4] --class com.intel.analytics.bigdl.finetune_flickr_style.Test dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar -f ~/data/flickr_style --model ~/model/model.5000 --node 1 --core 28 --env spark -b 224
```
Spark cluster mode, example command
```
./dist/bin/bigdl.sh -- spark-submit --class com.intel.analytics.bigdl.example.finetune_flickr_style.Test dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar -f hdfs://data/flickr_style --model ~/model/model.5000 --node 4 --core 4 --env spark -b 1024
```

## Conclusion
From finetune pre-trained model, we get speed up the training the different  Flickr images dataset with different style recognition. 