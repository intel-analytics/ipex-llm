**Project Orca: Easily Scaling out Python AI pipelines**

Most AI projects start with a Python notebook running on a single laptop; however, one usually needs to go through a mountain of pains to scale it to handle larger data set in a distributed fashion. 

_Project Orca_ allows you to easily scale out your single node Python notebook across large clusters, by providing:

* Data-parallel preprocessing for Python AI (supporting common Python libraries such as Pandas, Numpy, PIL, TensorFlow Dataset, PyTorch DataLoader, etc.)

* Sklearn-style APIs for transparently distributed training and inference (supporting TensorFlow, PyTorch, Keras, MXNet, Horovod, etc.)
