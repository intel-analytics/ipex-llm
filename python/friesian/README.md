# BigDL Friesian

BigDL Friesian is an application framework for building optimized large-scale recommender solutions. The recommending workflows built on top of Friesian can seamlessly scale out
to distributed big data in production on Xeon clusters. 

Friesian provides end-to-end support for three typical stages in a production recommendation system:

- Offline stage: distributed feature engineering and model training.
- Nearline stage: Feature and model updates.
- Online stage: Recall and ranking.

The overall architecture of Friesian is as follows:

<img src="../../scala/friesian/src/main/resources/images/architecture.png" width="800" />

See [here](./example) for the uses cases of various recommendation models implemented in Friesian.

See [here](../../scala/friesian) for the end-to-end serving pipeline in Friesian.
