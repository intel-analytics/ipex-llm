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

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{MiniBatch, PaddingParam, Sample, SampleToMiniBatch, Transformer, DataSet => _}
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.{Graph, StaticGraph}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.nn.mkldnn.{DnnGraph, MklDnnContainer}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.{DistributedImageFrame, ImageFeature, ImageFrame}
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.utils.intermediate.{ConversionUtils, IRGraph}
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

object Predictor {
  def apply[T: ClassTag](model: Module[T],
    featurePaddingParam: Option[PaddingParam[T]] = None,
    batchPerPartition: Int = 4)
    (implicit ev: TensorNumeric[T]): Predictor[T] = {
    new Predictor[T](model, featurePaddingParam, batchPerPartition)
  }

  private[optim] def predictImageBatch[T: ClassTag](
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

  private[optim] def predictSamples[T: ClassTag]
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

  private[optim] def splitTensor[T: ClassTag](output: Tensor[T],
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

  private[optim] def splitBatch[T: ClassTag](output: Activity, shareBuffer: Boolean, batchSize: Int)
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
    val dummyInput = getDummyData(imageFrame.rdd, batchPerPartition)
    val totalBatch = imageFrame.rdd.partitions.length * batchPerPartition
    val rdd = ConversionUtils.coalesce(imageFrame.asInstanceOf[DistributedImageFrame].rdd)
    val modelBroad = ModelBroadcast[T]().broadcast(rdd.sparkContext,
      ConversionUtils.convert(model.evaluate()), dummyInput)

    val realPartitionLength = rdd.partitions.length
    val toBatchBroad = rdd.sparkContext.broadcast(SampleToMiniBatch(
      batchSize = totalBatch,
      partitionNum = Some(realPartitionLength),
      featurePaddingParam = featurePaddingParam), shareBuffer)
    val localBatchPerPartition = totalBatch / realPartitionLength

    val result = rdd.mapPartitions(partition => {
      val localModel = modelBroad.value(false, true, dummyInput)
      val localToBatch = toBatchBroad.value._1.cloneTransformer()
      partition.grouped(localBatchPerPartition).flatMap(imageFeatures => {
        Predictor.predictImageBatch[T](localModel, imageFeatures, outputLayer, predictKey,
          localToBatch, shareBuffer)
        imageFeatures
      })
    })
    ImageFrame.rdd(result)
  }

  def predict[T: ClassTag](dataSet: RDD[Sample[T]], batchSize: Int = -1,
    shareBuffer: Boolean = false, model: Module[T], batchPerPartition: Int,
    featurePaddingParam: Option[PaddingParam[T]])(implicit ev: TensorNumeric[T]): RDD[Activity] = {
    val partitionNum = dataSet.partitions.length
    val totalBatch = if (batchSize > 0) {
      require(batchSize % partitionNum == 0, s"Predictor.predict: total batch size $batchSize " +
        s"should be divided by partitionNum ${partitionNum}")
      batchSize
    } else {
      batchPerPartition * partitionNum
    }
    val dummyInput = getDummyData(dataSet, totalBatch / partitionNum)
    val modelBroad = ModelBroadcast[T]().broadcast(dataSet.sparkContext,
      ConversionUtils.convert(model.evaluate()), dummyInput)
    val rdd = ConversionUtils.coalesce(dataSet)
    val otherBroad = rdd.sparkContext.broadcast(SampleToMiniBatch(
      batchSize = totalBatch,
      partitionNum = Some(rdd.partitions.length),
      featurePaddingParam = featurePaddingParam))
    rdd.mapPartitions { partition =>
      val localModel = modelBroad.value(false, true, dummyInput)
      val localTransformer = otherBroad.value.cloneTransformer()
      val miniBatch = localTransformer(partition)
      miniBatch.flatMap(batch => {
        val output = localModel.forward(batch.getInput)
        splitBatch(output, shareBuffer, batch.size())
      })
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

  // because Evaluator will use it too, we extend the scope out of Predictor
  private[optim] def getDummyData[T: ClassTag, R](dataSet: RDD[R],
    batchSize: Int)(implicit ev: TensorNumeric[T]): Activity = {
    if (Engine.getEngineType() == MklDnn && Engine.isMultiModels) {
      // here has an assumption, batchSizePerPar is not very large.
      val samples = dataSet.takeSample(withReplacement = false, num = batchSize)
        .map {
          case feature: ImageFeature => feature[Sample[T]](ImageFeature.sample)
          case sample => sample.asInstanceOf[Sample[T]]
        }
      val sampleToMiniBatch = SampleToMiniBatch(batchSize, partitionNum = Some(1))
      val miniBatch = sampleToMiniBatch(samples.toIterator).toSeq
      miniBatch.head.getInput()
    } else {
      Tensor()
    }
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
class Predictor[T: ClassTag] private[optim](
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
