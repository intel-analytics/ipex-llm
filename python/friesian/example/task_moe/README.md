# End-to-end Multi-task Recommender System with Verizon

This is the joint work with __Verizon__ to build an end-to-end multi-task recommender system for Personalization AI(PZAI).

We implement the task-level Mixture-of-Experts([task-MoE](https://arxiv.org/abs/2110.03742)) for recommendation tasks in our system.
We also leverage state-of-the-art architectures including Parallel Transformer, BST(https://arxiv.org/pdf/1905.06874.pdf) and TxT(https://arxiv.org/pdf/2010.06197.pdf), etc.

Our system provides end-to-end support for:

- Big data ETL: feature extraction pipeline based on Spark
- Model Training: large scale distributed model training on BigDL, Ray and Pytorch Lightning
- Serving: High performance model serving on seldon-core and Ray, support Python/Java/Scala API
