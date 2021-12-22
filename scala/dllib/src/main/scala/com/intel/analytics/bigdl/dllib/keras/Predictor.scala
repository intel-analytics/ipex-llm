/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.dllib.keras

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dllib._
import com.intel.analytics.bigdl.dllib.feature.common.{FeatureLabelPreprocessing, Preprocessing, ScalarToTensor, SeqToTensor}
import com.intel.analytics.bigdl.dllib.feature.dataset._
import com.intel.analytics.bigdl.dllib.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.optim.LocalPredictor
import com.intel.analytics.bigdl.dllib.tensor.{DoubleType, FloatType, Tensor}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.{DistributedImageFrame, ImageFeature, ImageFrame, LocalImageFrame}
import com.intel.analytics.bigdl.dllib.utils.{Engine, T, Table}
import com.intel.analytics.bigdl.dllib.feature.image.ImageSet
import com.intel.analytics.bigdl.dllib.feature.text._
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils
import org.apache.spark.ml.VectorCompatibility
import org.apache.spark.ml.adapter.SchemaUtils
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{ArrayType, DataType, DoubleType => sqlDoubleType, FloatType => sqlFloatType}
import org.apache.spark.sql.{DataFrame, Row}

import scala.collection.Iterator
import scala.reflect.ClassTag


object Predictor {
  def apply[T: ClassTag](model: Module[T],
                         featurePaddingParam: Option[PaddingParam[T]] = None,
                         batchPerPartition: Int = 4)
                        (implicit ev: TensorNumeric[T]): Predictor[T] = {
    new Predictor[T](model, featurePaddingParam, batchPerPartition)
  }

  private[bigdl] def predictImageBatch[T: ClassTag](
                     localModel: Module[T], imageFeatures: Seq[ImageFeature],
                     outputLayer: String, predictKey: String,
                     localToBatch: Transformer[Sample[T], MiniBatch[T]],
                     shareBuffer: Boolean)(implicit ev: TensorNumeric[T]): Seq[ImageFeature] = {
    val validImageFeatures = imageFeatures.filter(_.isValid)
    val samples = validImageFeatures.map(x => x[Sample[T]](ImageFeature.sample))
    val batchOut = predictSamples(localModel, samples, localToBatch, shareBuffer, outputLayer)
    validImageFeatures.toIterator.zip(batchOut).foreach(tuple => {
      tuple._1(predictKey) = tuple._2
    })
    imageFeatures
  }

  private[bigdl] def predictSamples[T: ClassTag]
  (localModel: Module[T], samples: Seq[Sample[T]],
   localToBatch: Transformer[Sample[T], MiniBatch[T]],
   shareBuffer: Boolean,
   outputLayer: String = null)(implicit ev: TensorNumeric[T]): Iterator[Activity] = {
    val layer = if (outputLayer == null) {
      localModel
    } else {
      val ol = localModel(outputLayer)
      require(ol.isDefined, s"cannot find layer that map name $outputLayer")
      ol.get
    }
    localToBatch(samples.toIterator).flatMap(batch => {
      localModel.forward(batch.getInput())
      splitBatch[T](layer.output, shareBuffer, batch.size())
    })
  }

  private[bigdl] def splitTensor[T: ClassTag](output: Tensor[T],
                                              shareBuffer: Boolean, batchSize: Int)
                                             (implicit ev: TensorNumeric[T]): Array[Activity] = {
    val result = if (shareBuffer) output else output.clone
    val out = if (batchSize == 1) {
      Array(result.squeeze)
    } else {
      val size = result.size(1)
      require(batchSize == size,
        s"The batchSize is required to be $size, while actual is $batchSize")
      result.split(1)
    }
    out.asInstanceOf[Array[Activity]]
  }

  private[bigdl] def splitBatch[T: ClassTag](output: Activity, shareBuffer: Boolean, batchSize: Int)
                                            (implicit ev: TensorNumeric[T]): Array[Activity] = {
    val out = if (output.isTensor) {
      splitTensor(output.toTensor, shareBuffer, batchSize)
    } else {
      val result = output.toTable
      val tables = new Array[Table](batchSize)


      (1 to result.length()).foreach(key => {
        val split = splitBatch(result(key), shareBuffer, batchSize)
        val size = split.length
        require(batchSize == size,
          s"The batchSize is required to be $size, while actual is $batchSize")
        var i = 0
        while (i < batchSize) {
          if (tables(i) == null) tables(i) = T()
          tables(i).insert(split(i))
          i += 1
        }
      })
      tables
    }
    out.asInstanceOf[Array[Activity]]
  }


  def predictImage[T: ClassTag](imageFrame: DistributedImageFrame,
                                outputLayer: String = null,
                                shareBuffer: Boolean = false,
                                predictKey: String = ImageFeature.predict,
                                batchPerPartition: Int,
                                model: Module[T],
                                featurePaddingParam: Option[PaddingParam[T]])(
                                 implicit ev: TensorNumeric[T]): DistributedImageFrame = {
    val localBatchPerPartition = batchPerPartition

    val rdd = imageFrame.asInstanceOf[DistributedImageFrame].rdd
    val modelBroad = ModelBroadcast[T]().broadcast(rdd.sparkContext, model)
    val partitionNum = rdd.partitions.length
    val toBatchBroad = rdd.sparkContext.broadcast(SampleToMiniBatch(
      batchSize = partitionNum * batchPerPartition,
      partitionNum = Some(partitionNum),
      featurePaddingParam = featurePaddingParam), shareBuffer)
    val result = rdd.mapPartitions(partition => {
      val localModel = modelBroad.value()
      localModel.evaluate()
      val localToBatch = toBatchBroad.value._1.cloneTransformer()
      val batchedIter = partition.grouped(localBatchPerPartition) ++ Array(null)
      batchedIter.flatMap { imageFeatures =>
        if (imageFeatures != null ) {
          Predictor.predictImageBatch[T](localModel, imageFeatures, outputLayer, predictKey,
            localToBatch, shareBuffer)
          imageFeatures
        } else {
          localModel.release()
          Seq.empty
        }
      }
    })
    ImageFrame.rdd(result)
  }

  def predict[T: ClassTag](dataSet: RDD[Sample[T]], batchSize: Int = -1,
      shareBuffer: Boolean = false, model: Module[T], batchPerPartition: Int,
      featurePaddingParam: Option[PaddingParam[T]])
                          (implicit ev: TensorNumeric[T]): RDD[Activity] = {
    val modelBroad = ModelBroadcast[T]().broadcast(dataSet.sparkContext, model)
    val partitionNum = dataSet.partitions.length
    val totalBatch = if (batchSize > 0) {
      require(batchSize % partitionNum == 0, s"Predictor.predict: total batch size $batchSize " +
        s"should be divided by partitionNum ${partitionNum}")
      batchSize
    } else {
      batchPerPartition * partitionNum
    }
    val otherBroad = dataSet.sparkContext.broadcast(SampleToMiniBatch(
      batchSize = totalBatch,
      partitionNum = Some(partitionNum),
      featurePaddingParam = featurePaddingParam))
    dataSet.mapPartitions { partition =>
      val localModel = modelBroad.value()
      localModel.evaluate()
      val localTransformer = otherBroad.value.cloneTransformer()
      val miniBatch = localTransformer(partition) ++ Array(null)
      miniBatch.flatMap { batch =>
        if (batch != null) {
          val output = localModel.forward(batch.getInput)
          splitBatch(output, shareBuffer, batch.size())
        } else {
          localModel.release()
          Seq.empty
        }
      }
    }
  }


    def predictMiniBath[T: ClassTag](dataSet: RDD[MiniBatch[T]],
                                     model: Module[T])
                            (implicit ev: TensorNumeric[T]): RDD[Activity] = {
      val modelBroad = ModelBroadcast[T]().broadcast(dataSet.sparkContext, model)
      dataSet.mapPartitions { partition =>
        val localModel = modelBroad.value()
        localModel.evaluate()
        val miniBatch = partition ++ Array(null)
        miniBatch.flatMap { batch =>
          if (batch != null) {
            val output = localModel.forward(batch.getInput)
            splitBatch(output, false, batch.size())
          } else {
            localModel.release()
            Seq.empty
          }
        }
      }
  }

  def predictClass[T: ClassTag](dataSet: RDD[Sample[T]], batchSize: Int = -1, model: Module[T],
             batchPerPartition: Int, featurePaddingParam: Option[PaddingParam[T]])(
    implicit ev: TensorNumeric[T]): RDD[Int] = {
    val result = Predictor.predict(dataSet, batchSize, true, model,
      batchPerPartition, featurePaddingParam)
    result.mapPartitions { partition =>
      partition.map(output => {
        val _output = output.toTensor[T]
        require(_output.dim() == 1, s"Predictor.predictClass:" +
          s"Only support one sample has one label, but got ${_output.dim()} label")
        ev.toType[Int](_output.max(1)._2.valueAt(1))
      })
    }
  }
}

trait Predictable[T] extends VectorCompatibility{

  protected val module: Module[T]

  implicit val tag: ClassTag[T]
  implicit val ev: com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric[T]


  /**
   * Use a model to do prediction for RDD.
   *
   * @param x Prediction data, RDD of Sample.
   * @param batchPerThread The total batchSize is batchPerThread * rdd.getNumPartitions.
   */
  def predict(
               x: RDD[Sample[T]],
               batchPerThread: Int)(implicit ev: TensorNumeric[T]): RDD[Activity] = {
    Predictor.predict(x,
      batchSize = -1,
      shareBuffer = false,
      model = module,
      batchPerPartition = batchPerThread,
      featurePaddingParam = None)
  }

  /**
   * Use a model to do prediction for RDD.
   * The default batchPerThread is 4,
   * and the total batchSize is batchPerThread * rdd.getNumPartitions.
   * @param x Prediction data, RDD of Sample.
   */
  def predict(
               x: RDD[Sample[T]])(implicit ev: TensorNumeric[T]): RDD[Activity] = {
    predict(x, batchPerThread = 4)
  }

  def predictMiniBatch(x: RDD[MiniBatch[T]])(implicit ev: TensorNumeric[T]): RDD[Activity] = {
    Predictor.predictMiniBath(x, model = module)
  }

  /**
   * Use a model to do prediction in local mode.
   *
   * @param x Prediction data, LocalDataSet.
   * @param batchPerThread The total batchSize is batchPerThread * numOfCores.
   */
  def predict(
               x: LocalDataSet[MiniBatch[T]],
               batchPerThread: Int)(implicit ev: TensorNumeric[T]): Array[Activity] = {
    val localPredictor = LocalPredictor(module, batchPerCore = batchPerThread)
    val result = localPredictor.predict(x)
    localPredictor.shutdown()
    result
  }

  /**
   * Use a model to do prediction in local mode.
   * The total batch size is batchPerThread * numOfCores, and batchPerThread is 4 by default.
   * @param x Prediction data, LocalDataSet.
   */
  def predict(
               x: LocalDataSet[MiniBatch[T]])(implicit ev: TensorNumeric[T]): Array[Activity] = {
    predict(x, batchPerThread = 4)
  }

  /**
   * Use a model to do prediction in local mode.
   *
   * @param x Prediction data, array of Sample.
   * @param batchPerThread The total batchSize is batchPerThread * numOfCores.
   */
  def predict(
               x: Array[Sample[T]],
               batchPerThread: Int)(implicit ev: TensorNumeric[T]): Array[Activity] = {
    val localPredictor = LocalPredictor(module, batchPerCore = batchPerThread)
    val result = localPredictor.predict(x)
    localPredictor.shutdown()
    result
  }

  /**
   * Use a model to do prediction in local mode.
   * The total batch size is batchPerThread * numOfCores, and batchPerThread is 4 by default.
   * @param x Prediction data, array of Sample.
   */
  def predict(
               x: Array[Sample[T]])(implicit ev: TensorNumeric[T]): Array[Activity] = {
    predict(x, batchPerThread = 4)
  }

  /**
   * Use a model to do prediction on ImageSet.
   *
   * @param x Prediction data, ImageSet.
   * @param batchPerThread The total batch size is
   *        batchPerThread * rdd.getNumPartitions(distributed mode)
   *        or batchPerThread * numOfCores(local mode)
   */
  def predict(
               x: ImageSet,
               batchPerThread: Int): ImageSet = {

    val resultImageFrame = x.toImageFrame() match {
      case distributedImageFrame: DistributedImageFrame =>
        Predictor(module, None, batchPerThread)
          .predictImage(distributedImageFrame, outputLayer = null)
      case localImageFrame: LocalImageFrame =>
        val predictor = LocalPredictor(module, None, batchPerCore = batchPerThread)
        val imageFrame = predictor.predictImage(localImageFrame, outputLayer = null,
          shareBuffer = false)
        predictor.shutdown()
        imageFrame
    }
    ImageSet.fromImageFrame(resultImageFrame)
  }

  /**
   * The default batchPerThread is 4.
   * For DistributedImageSet, the total batchSize is batchPerThread * rdd.getNumPartitions.
   * For LocalImageSet, the total batchSize is batchPerThread * numOfCores.
   *
   * @param x Prediction data, ImageSet.
   */
  def predict(
               x: ImageSet): ImageSet = {
    predict(x, batchPerThread = 4)
  }

  def unwrapVectorAsNecessary(colType: DataType): (Row, Int) => Any = {
    // to support both ML Vector and MLlib Vector
    if (colType.typeName.contains("vector")) {
      (row: Row, index: Int) => getVectorSeq(row, colType, index)
    } else {
      (row: Row, index: Int) => row.get(index)
    }
  }

  var predictTransformer: Preprocessing[Any, Sample[T]] = null

  def outputToPrediction(output: Tensor[T]): Any = {
    output.clone().storage().array()
  }

  def predict(x: DataFrame,
              featureCols: Array[String],
              predictionCol: String,
              batchPerThread: Int): DataFrame = {
    require(predictTransformer!=null, "Must train the model before call predcition")

    val guid = java.util.UUID.randomUUID.toString
    val internalFeatureCol = "features" + guid
    val df = if (featureCols.size > 1) {
      val assembler = new VectorAssembler()
        .setInputCols(featureCols)
        .setOutputCol(internalFeatureCol)
      assembler.transform(x)
    } else {
      x.withColumnRenamed(featureCols.head, internalFeatureCol)
    }

    val featureColIndex = df.schema.fieldIndex(internalFeatureCol)
    val featureType = df.schema(internalFeatureCol).dataType
    val featureFunc = unwrapVectorAsNecessary(featureType)

    val sc = x.sqlContext.sparkContext
    val modelBroadCast = ModelBroadcast[T]().broadcast(sc, module.evaluate())

    val featureTransformersBC = sc.broadcast(predictTransformer)
    val toBatchBC = sc.broadcast(SampleToMiniBatch[T](batchPerThread, partitionNum = Some(1)))

    // concat the prediction and other columns in DF. avoid zip between RDD
    val resultRDD = df.rdd.mapPartitions { rowIter =>
      val localModel = modelBroadCast.value()
      localModel.evaluate()
      val featureSteps = featureTransformersBC.value.cloneTransformer()
      val toBatch = toBatchBC.value.cloneTransformer()

      rowIter.grouped(batchPerThread).flatMap { rowBatch =>
        val featureSeq = rowBatch.map(r => featureFunc(r, featureColIndex))
        val samples = featureSteps(featureSeq.iterator)
        val predictions = toBatch(samples).flatMap { batch =>
          val batchResult = localModel.forward(batch.getInput()).toTensor
          if (batchResult.size().length == 2) {
            batchResult.split(1).map(outputToPrediction)
          } else if (batchResult.size().length == 1) {
            Array(outputToPrediction(batchResult))
          } else {
            throw new RuntimeException(
              "unexpected batchResult dimension: " + batchResult.size().mkString(", "))
          }
        }
        rowBatch.toIterator.zip(predictions).map { case (row, predict) =>
          Row.fromSeq(row.toSeq ++ Seq(predict))
        }
      }
    }

    val resultSchema =
      ev.getType() match {
      case DoubleType =>
        SchemaUtils.appendColumn(df.schema, predictionCol, ArrayType(sqlDoubleType, false))
      case FloatType =>
        SchemaUtils.appendColumn(df.schema, predictionCol, ArrayType(sqlFloatType, false))
      case _ => throw new Exception("Only support Double and Float for now")
    }

    x.sqlContext.createDataFrame(resultRDD, resultSchema)
  }

  def predict(x: DataFrame,
              featureCols: Array[String],
              predictionCol: String): DataFrame = {
    predict(x, featureCols, predictionCol, batchPerThread = 4)
  }

  /**
   * Use a model to do prediction on TextSet.
   *
   * @param x Prediction data, TextSet.
   * @param batchPerThread The total batch size is
   *        batchPerThread * rdd.getNumPartitions(distributed mode)
   *        or batchPerThread * numOfCores(local mode)
   */
  def predict(
               x: TextSet,
               batchPerThread: Int): TextSet = {
    x match {
      case distributed: DistributedTextSet =>
        TextPredictor[T](module, batchPerThread).predict(distributed)
      case local: LocalTextSet =>
        val features = local.array
        val samples = features.map(_.getSample).asInstanceOf[Array[Sample[T]]]
        val predictions = predict(samples, batchPerThread)
        val results = features.zip(predictions).map{case (feature, predict) =>
          feature(TextFeature.predict) = predict
          feature
        }
        TextSet.array(results).setWordIndex(x.getWordIndex)
    }
  }

  /**
   * The default batchPerThread is 4.
   * For DistributedTextSet, the total batchSize is batchPerThread * rdd.getNumPartitions.
   * For LocalTextSet, the total batchSize is batchPerThread * numOfCores.
   *
   * @param x Prediction data, TextSet.
   */
  def predict(
               x: TextSet): TextSet = {
    predict(x, batchPerThread = 4)
  }

  /**
   * Use a model to predict for classes. By default, label predictions start from 0.
   *
   * @param x Prediction data, RDD of Sample.
   * @param batchPerThread The default batchPerThread is 4,
   *       and the total batchSize is batchPerThread * rdd.getNumPartitions.
   * @param zeroBasedLabel Boolean. Whether result labels start from 0.
   *                       Default is true. If false, result labels start from 1.
   */
  def predictClasses(
                      x: RDD[Sample[T]],
                      batchPerThread: Int = 4,
                      zeroBasedLabel: Boolean = true): RDD[Int] = {
    com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils.toZeroBasedLabel(zeroBasedLabel,
      Predictor.predictClass(x,
        batchSize = -1,
        model = module,
        batchPerPartition = batchPerThread,
        featurePaddingParam = None))
  }

}

/**
 * Predictor for distributed data
 *
 * NOTE: The `predictClass`, `predict` and `predictImage` will call the relevant methods of
 * object `Predictor`. Why we do this? Because every these methods uses the ClassTag `T`. If we do
 * these jobs in the methods of class`Predictor`, when we do `mapPartition`, Spark will find all
 * used values and do serialization. The `T` is the argument of constructor, the serialization will
 * package the whole `Predictor` class, which contains the`model`. It will send a duplicate model
 * to the workers. So we should move these methods to object `Predictor`.
 *
 * @param model BigDL model
 * @param featurePaddingParam featurePaddingParam if the inputs have variant size
 * @param batchPerPartition batch size per partition, default is 4
 */
class Predictor[T: ClassTag] private[bigdl](
                                             model: Module[T],
                                             featurePaddingParam: Option[PaddingParam[T]] = None,
                                             batchPerPartition: Int = 4)
                                           (implicit ev: TensorNumeric[T]) extends Serializable {

  def predictClass(dataSet: RDD[Sample[T]], batchSize: Int = -1): RDD[Int] = {
    Predictor.predictClass(dataSet, batchSize, model, batchPerPartition, featurePaddingParam)
  }

  def predict(dataSet: RDD[Sample[T]], batchSize: Int = -1,
              shareBuffer: Boolean = false): RDD[Activity] = {
    Predictor.predict(dataSet, batchSize, shareBuffer, model, batchPerPartition,
      featurePaddingParam)
  }


  /**
   * model predict DistributedImageFrame, return imageFrame with predicted tensor
   * @param imageFrame imageFrame that contains images
   * @param outputLayer if outputLayer is not null, the output of layer that matches
   *                      outputLayer will be used as predicted output
   * @param shareBuffer whether to share same memory for each batch predict results
   * @param predictKey key to store predicted result
   */
  def predictImage(imageFrame: DistributedImageFrame,
                   outputLayer: String = null,
                   shareBuffer: Boolean = false,
                   predictKey: String = ImageFeature.predict): DistributedImageFrame = {
    Predictor.predictImage(imageFrame, outputLayer, shareBuffer, predictKey, batchPerPartition,
      model, featurePaddingParam)
  }
}
