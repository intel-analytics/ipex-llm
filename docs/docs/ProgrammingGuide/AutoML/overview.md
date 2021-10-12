
## AutoML Framework Overview

There are four essential components in the AutoML framework, i.e. FeatureTransformer, Model, SearchEngine, and Pipeline. 

A FeatureTransformer (inherited from _BaseFeatureTransformer_ class) defines the feature engineering process, which usually includes a chain of operations like feature generation, feature transformations and selection. A Model (inherited from _BaseModel_ class) usually defines an optimizable model (e.g. AlexNet or LeNet), and a fitting function using an optimization algorithm (e.g. SGD, Adam, etc.). A Model may also include the procedure of model/algorithm selection. 

During training, a SearchEngine (inherited from _SearchEngine_ class) searches for the best set of hyper parameters for both FeatureTransformer and Model and control the actual model fitting process. A Pipeline (inherited from _Pipeline_ class) is a convenient utility that integrates FeatureTransformer and Model into an end2end data processing pipeline. A Pipeline can be easily saved to file and loaded for reuse later elsewhere. 

A typical training workflow with AutoML looks like below: 

1.	A FeatureTransformer and A Model are instantiated. A SearchEngine is then instantiated and configured with the FeatureTransformer and Model, along with search presets, specifying how parameters are searched, the reward metric, and etc. 
2.	The SearchEngine runs the search procedure. It will generate several trials at a time and distribute the trials in a cluster. Each trail runs feature engineering and the model fitting process with a different combination of hyper parameters and obtain the target metric. It may take a while if the search presets generate many trails or model fitting takes a long time.
3.	After all trials completed, the best configuration and fitted model are retrieved according to the target metric. They are used to generate the result FeatureTransformer and Model, which are in turn used to compose a Pipeline.  The Pipeline can then be saved to file and loaded later for inference and resume/incremental training. 

