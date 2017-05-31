#Tensorflow protobuf Java classes
This folder contains Tensorflow protobuf classes. So we can provide some features like 
fine-tune/inference saved Tensorflow models on Spark and save BigDL model in format which can be 
loaded by Tensorflow(e.g. for inference on mobile).

##Why not use Tensorflow java API
We cannot just import Tensorflow java API jar from maven and use it. Tensorflow must be installed on
the machine. This brings unnecessary dependency.

Tensorflow Java API is not so sufficient to parse the model graph.

##Which version of Tensorflow are these codes generate from?
Tensorflow 1.0.0

##How to generate the classes?
Download protobuf binary from [here](https://github.com/google/protobuf/releases/download/v3.0.2/protoc-3.0.2-linux-x86_64.zip).

After extract the package, go to the bin folder and run
```bash
protoc -I=$path_to_tensorflow --java_out=./ $path_to_tensorflow/tensorflow/core/framework/*.proto
```

Then you can see the generated Java class files in the current folder.