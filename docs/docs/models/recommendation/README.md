# Analytics Zoo Recommender API
Analytics Zoo provides two Recommenders, including wide and deep(WND) model and Neural network-based Collaborative Filtering(NCF) model. This model could be fed into NNframes and BigDL Optimizer directly.
Recommender can handle models with explict/implicit feedback, given corresponding features. It also provide 3 user-friendly APIs to predict user item pairs, and recommend items(users) for users(items). 

## Wide and deep
WND Learning Model, proposed by ([Google, 2016](https://arxiv.org/pdf/1606.07792.pdf)), is a DNN-Linear mixed model. WND combines the strength of memorization and generalization. It's useful for generic large-scale regression and classification problems with sparse input features(e.g., categorical features with a large number of possible feature values). It has been used for Google App Store for their app recommendation.
[Scala example](https://github.com/intel-analytics/zoo/blob/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/recommendation/WideAndDeepExample.scala)

It's very easy to build a WND model for recommendation with below code piece. 

```scala
// define feature column information according to the data for feature engineering and model
val localColumnInfo = ColumnFeatureInfo("...")
// build a WND model
val wideAndDeep: WideAndDeep[Float] = WideAndDeep[Float](
      params.modelType,  // modelType of "wide_n_deep", "wide", and "deep" are supported
      numClasses = params.numClasses,
      columnInfo = localColumnInfo)
```

After training the model, users can predict user item pairs, and recommend items(users) for users(items) given a RDD of UserItemFeature, which includes user item-pair candidates and corresponding features.

```scala
val userItemPairPrediction = wideAndDeep.predictUserItemPair(validationpairFeatureRdds)
val userRecs = wideAndDeep.recommendForUser(validationpairFeatureRdds, 3)
val itemRecs = wideAndDeep.recommendForItem(validationpairFeatureRdds, 3)
```

## Neural network-based Collaborative Filtering
NCF([He, 2015](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf)) leverages a multi-layer perceptrons to learn the user–item interaction function, at the mean time, NCF can express and generalize matrix factorization under its framework. includeMF(Boolean) is provided for users to build a NCF with or without matrix factorization. 
[Scala example](https://github.com/intel-analytics/zoo/blob/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/recommendation/NeuralCFexample.scala)

Users can build a NCF model using below code piece. 

```scala
val ncf = NeuralCF[Float](
      userCount = userCount,
      itemCount = itemCount,
      numClasses = params.numClasses ,
      userEmbed = params.userEmbed,
      itemEmbed = params.itemEmbed,
      includeMF = true,
      hiddenLayers = Array(40, 20, 10))
```

After training the model, users can predict user item pairs, and recommend items(users) for users(items) given a RDD of UserItemFeature, which includes user item-pair candidates and corresponding features.

```scala
val userItemPairPrediction = ncf.predictUserItemPair(validationpairFeatureRdds)
val userRecs = ncf.recommendForUser(validationpairFeatureRdds, 3)
val itemRecs = ncf.recommendForItem(validationpairFeatureRdds, 3)
```