# DLlib Quickstarts

---

![](../../../../image/colab_logo_32px.png)[Run in Google Colab](https://colab.research.google.com/github/intel-analytics/BigDL/blob/branch-2.0/python/dllib/colab-notebook/dllib_keras_api.ipynb) &nbsp;![](../../../../image/GitHub-Mark-32px.png)[View source on GitHub](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/dllib/colab-notebook/dllib_keras_api.ipynb)

---

**In this guide we will demonstrate how to use _DLlib keras style api_ and _DLlib NNClassifier_ for classification.**

### **Step 0: Prepare Environment**

We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the environment. Please refer to the [install guide](../Overview/chronos.html#install) for more details.

```bash
conda create -n my_env python=3.7 # "my_env" is conda environment name, you can use any name you like.
conda activate my_env
pip install bigdl-dllib
```

### Step 1: Data loading and processing using Spark DataFrame

```python
df = spark.read.csv(path, sep=',', inferSchema=True).toDF("num_times_pregrant", "plasma_glucose", "blood_pressure", "skin_fold_thickness", "2-hour_insulin", "body_mass_index", "diabetes_pedigree_function", "age", "class")
```

We process the data using Spark API and split the data into train and test set.

```python
vecAssembler = VectorAssembler(outputCol="features")
vecAssembler.setInputCols(["num_times_pregrant", "plasma_glucose", "blood_pressure", "skin_fold_thickness", "2-hour_insulin", "body_mass_index", "diabetes_pedigree_function", "age"])
train_df = vecAssembler.transform(df)

changedTypedf = train_df.withColumn("label", train_df["class"].cast(DoubleType())+lit(1))\
    .select("features", "label")
(trainingDF, validationDF) = changedTypedf.randomSplit([0.9, 0.1])
```

### Step 3: Define classification model using DLlib keras style api

```python
x1 = Input(shape=(8,))
dense1 = Dense(12, activation='relu')(x1)
dense2 = Dense(8, activation='relu')(dense1)
dense3 = Dense(2)(dense2)
model = Model(x1, dense3)
```

### Step 4: Create NNClassifier and Fit NNClassifier

```python
classifier = NNClassifier(model, CrossEntropyCriterion(), [8]) \
    .setOptimMethod(Adam()) \
    .setBatchSize(32) \
    .setMaxEpoch(150)

nnModel = classifier.fit(trainingDF)
```

### Step 5: Evaluate the trained model

```python
predictionDF = nnModel.transform(validationDF).cache()
predictionDF.sample(False, 0.1).show()


evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictionDF)
```
