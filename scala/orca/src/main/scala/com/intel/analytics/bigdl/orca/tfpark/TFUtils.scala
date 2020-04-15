/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.tfpark

import java.io.{File, FileInputStream, InputStream}
import java.nio._

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractCriterion, AbstractModule, Activity}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.zoo.feature.common.Preprocessing
import com.intel.analytics.zoo.feature.image.ImageProcessing
import com.intel.analytics.zoo.pipeline.api.keras.{metrics => kmetrics}
import com.intel.analytics.zoo.pipeline.api.keras.metrics.{Accuracy, BinaryAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy}
import org.tensorflow.framework.GraphDef
import org.tensorflow.{DataType, Tensor => TTensor}

import scala.io.Source
import scala.reflect.io.Path

object TFUtils {

  val defaultSessionConfig = SessionConfig()

  private[zoo] def getTrainMeta(trainMetaPath: Path) = {
    val jsonStr = Source.fromFile(trainMetaPath.jfile).getLines().mkString
    import org.json4s._
    import org.json4s.jackson.JsonMethods._
    implicit val formats = DefaultFormats

    parse(jsonStr).camelizeKeys.extract[TrainMeta]
  }

  private[zoo] def parseGraph(graphProtoTxt: String): GraphDef = {
    var fr: File = null
    var in: InputStream = null
    try {
      fr = new File(graphProtoTxt)
      in = new FileInputStream(fr)

      GraphDef.parseFrom(in)
    } finally {
      if (in != null) in.close()
    }
  }

  private def decodeUVarInt64(bytes: Array[Byte], offset: Int): (Long, Int) = {
    var shift = 0
    var p = offset
    var result: Long = 0
    while (shift  <= 63 && p < bytes.length) {
      val b = bytes(p)
      p += 1
      if ((b & 128) != 0) {
        result |= ((b & 127) << shift)
      } else {
        result |= (b << shift)
        return (result, p)
      }

      shift += 7
    }
    (result, p)
  }

  private def getOffsets(buffer: ByteBuffer, numElem: Int): Array[Int] = {
    val offsetsBuffer = ByteBuffer.wrap(buffer.array()
      .slice(buffer.arrayOffset(), numElem * 8))
      .order(ByteOrder.nativeOrder())
      .asLongBuffer()
    val offsets = new Array[Long](numElem)
    offsetsBuffer.get(offsets)
    offsets.map(_.toInt)
  }

  private[zoo] def tf2bigdl(t: TTensor[_], output: Tensor[_]) = {
    val shape = t.shape().map(_.toInt)
    output.resize(shape)
    val dataType = t.dataType()

    val numericDataTypes = Set(DataType.FLOAT,
      DataType.UINT8, DataType.INT32, DataType.INT64, DataType.DOUBLE)

    if (dataType == DataType.STRING) {
      val outputTensor = output.asInstanceOf[Tensor[Array[Byte]]]
      require(t.numDimensions() <= 1, "only scalar or Vector string are supported")
      val elements = t.numElements()
      val buffer = ByteBuffer.allocate(t.numBytes())
      t.writeTo(buffer)
      val offsets = getOffsets(buffer, elements)
      val storage = outputTensor.storage().array()
      val strDataOffset = elements * 8
      var i = 0
      while (i < elements) {
        val offset = buffer.arrayOffset() + offsets(i) + strDataOffset
        val (strLen, strStart) = decodeUVarInt64(buffer.array(), offset)
        storage(outputTensor.storageOffset() - 1 + i) = buffer.array()
          .slice(strStart, strStart + strLen.toInt)
        i += 1
      }
    } else if (numericDataTypes(dataType)) {
      dataType match {
        case DataType.FLOAT =>
          val outputTensor = output.asInstanceOf[Tensor[Float]]
          val buffer = FloatBuffer.wrap(
            outputTensor.storage().array(),
            outputTensor.storageOffset() - 1,
            shape.product)
          t.writeTo(buffer)
        case DataType.UINT8 =>
          val outputTensor = output.asInstanceOf[Tensor[Float]]
          val arr = new Array[Byte](shape.product)
          val buffer = ByteBuffer.wrap(arr)
          t.writeTo(buffer)
          byte2float(arr, outputTensor.storage().array(), outputTensor.storageOffset() - 1)
        case DataType.INT32 =>
          val outputTensor = output.asInstanceOf[Tensor[Float]]
          val arr = new Array[Int](shape.product)
          val buffer = IntBuffer.wrap(arr)
          t.writeTo(buffer)
          int2float(arr, outputTensor.storage().array(), outputTensor.storageOffset() - 1)
        case DataType.INT64 =>
          val outputTensor = output.asInstanceOf[Tensor[Float]]
          val arr = new Array[Long](shape.product)
          val buffer = LongBuffer.wrap(arr)
          t.writeTo(buffer)
          long2float(arr, outputTensor.storage().array(), outputTensor.storageOffset() - 1)
        case DataType.DOUBLE =>
          val outputTensor = output.asInstanceOf[Tensor[Float]]
          val arr = new Array[Double](shape.product)
          val buffer = DoubleBuffer.wrap(arr)
          t.writeTo(buffer)
          double2float(arr, outputTensor.storage().array(), outputTensor.storageOffset() - 1)
      }

    } else {
      throw new Exception(s"data type ${dataType} are not supported")
    }
  }

  private[zoo] def byte2float(src: Array[Byte], dest: Array[Float], offset: Int): Unit = {
    val length = src.length
    var i = 0
    while (i < length) {
      dest(offset + i) = src(i).toFloat
      i += 1
    }
  }

  private[zoo] def int2float(src: Array[Int], dest: Array[Float], offset: Int): Unit = {
    val length = src.length
    var i = 0
    while (i < length) {
      dest(offset + i) = src(i).toFloat
      i += 1
    }
  }

  private[zoo] def long2float(src: Array[Long], dest: Array[Float], offset: Int): Unit = {
    val length = src.length
    var i = 0
    while (i < length) {
      dest(offset + i) = src(i).toFloat
      i += 1
    }
  }

  private[zoo] def double2float(src: Array[Double], dest: Array[Float], offset: Int): Unit = {
    val length = src.length
    var i = 0
    while (i < length) {
      dest(offset + i) = src(i).toFloat
      i += 1
    }
  }

  def tfenum2datatype(enum: Int): DataType = {
    enum match {
      case 1 => DataType.FLOAT
      case 2 => DataType.DOUBLE
      case 3 => DataType.INT32
      case 4 => DataType.UINT8
      case 7 => DataType.STRING
      case 9 => DataType.INT64
      case 10 => DataType.BOOL
      case _ => throw new IllegalArgumentException(s"unsupported tensorflow datatype $enum")

    }
  }

  def tfdatatype2enum(dataType: DataType): Int = {
    dataType match {
      case DataType.FLOAT => 1
      case DataType.DOUBLE => 2
      case DataType.INT32 => 3
      case DataType.UINT8 => 4
      case DataType.STRING => 7
      case DataType.INT64 => 9
      case DataType.BOOL => 10
      case _ => throw new IllegalArgumentException(s"unsupported tensorflow datatype $dataType")

    }
  }

}

class IdentityCriterion extends AbstractCriterion[Activity, Activity, Float]() {

  override def updateOutput(input: Activity, target: Activity): Float = {
    if (input.isTensor) {
      input.toTensor[Float].value()
    } else {
      val table = input.toTable
      table[Tensor[Float]](table.length()).value()
    }
  }
  override def updateGradInput(input: Activity, target: Activity): Activity = {
    gradInput
  }
}

class TFValidationMethod(val valMethod: ValidationMethod[Float],
                         name: String,
                         outputIndices: java.util.List[Int],
                         labelIndices: java.util.List[Int]) extends ValidationMethod[Float] {

  private def toActivity(indices: java.util.List[Int], table: Table) = {
    if (indices.size() == 1) {
      table[Tensor[Float]](indices.get(0) + 1)
    } else {
      var i = 0
      val outputs = T()
      while (i < indices.size()) {
        outputs.insert(table(indices.get(i) + 1))
        i += 1
      }
      outputs
    }
  }

  private def oneBasedLabel(activity: Activity) = {
    if (activity.isTensor) {
      activity.toTensor[Float].add(1.0f)
    } else {
      val t = activity.toTable
      var i = 0
      while (i < t.length()) {
        t[Tensor[Float]](i + 1).add(1.0f)
        i += 1
      }
    }
  }

  override def apply(output: Activity, target: Activity): ValidationResult = {
    // the output layout [grads..., outputs..., labels..., loss]
    val outputT = output.toTable

    if (valMethod.isInstanceOf[Loss[Float]]) {
      val loss = outputT[Tensor[Float]](outputT.length()).value()
      return new LossResult(loss, 1)
    }

    val outputActivity: Activity = toActivity(outputIndices, outputT)
    val targetActivity: Activity = toActivity(labelIndices, outputT)

    val to1basedLabel = valMethod match {
      case _: SparseCategoricalAccuracy[Float] => false
      case _: CategoricalAccuracy[Float] => false
      case _: BinaryAccuracy[Float] => false
      case v: kmetrics.Top5Accuracy[Float] => !v.zeroBasedLabel
      case v: Accuracy[Float] => !v.zeroBasedLabel
      case _: Top1Accuracy[Float] => true
      case _: Top5Accuracy[Float] => true
      case _: TreeNNAccuracy[Float] => true
      case _ => false
    }

    if (to1basedLabel) {
      oneBasedLabel(targetActivity)
    }

    valMethod.apply(outputActivity, targetActivity)
  }

  override protected def format(): String = {
    (name + " " + valMethod.toString()).trim
  }
}

class StatelessMetric(name: String, idx: Int) extends ValidationMethod[Float] {
  override def apply(output: Activity, target: Activity): ValidationResult = {
    // the output layout [grads..., metrics]
    val outputT = output.toTable

    val value = outputT[Tensor[Float]](idx + 1).value()
    val count = outputT[Tensor[Float]](outputT.length() - 1).value().toInt

    new ContiguousResult(value * count, count, name)
  }

  override protected def format(): String = {
    name
  }
}

class MergeFeatureLabel() extends ImageProcessing {

  def createNewMergedSample(sample: Sample[Float]): Sample[Float] = {
    val newSize = sample.getFeatureSize() ++ sample.getLabelSize()
    Sample(sample.getData(), newSize, null)
  }

  override def transform(feature: ImageFeature): ImageFeature = {
    val oldSample = feature[Sample[Float]](ImageFeature.sample)
    val newSample = createNewMergedSample(oldSample)
    val newFeature = new ImageFeature()
    newFeature(ImageFeature.sample) = newSample
    newFeature
  }
}

class MergeFeatureLabelFeatureTransformer() extends Preprocessing[Any, Any] {

  private val mergeFun = new MergeFeatureLabel()
  override def apply(prev: Iterator[Any]): Iterator[Any] = {
    prev.map(transform)
  }

  private def transform(element: Any): Any = {
    element match {
      case feature: ImageFeature =>
        mergeFun.transform(feature)
      case sample: Sample[Float] =>
        mergeFun.createNewMergedSample(sample)
      case _ => throw new IllegalArgumentException(
        s"Element type ImageFeaute and Sample[Float] is supported. " +
          s"Element type ${element.getClass} is not supported.")
    }
  }
}


case class TrainMeta(inputs: Array[String],
                     inputTypes: Array[Int],
                     additionalInputs: Array[String],
                     additionalInputTypes: Array[Int],
                     labels: Array[String],
                     labelTypes: Array[Int],
                     predictionOutputs: Array[String],
                     metricTensors: Array[String],
                     batchSizeTensor: String,
                     lossTensor: String,
                     variables: Array[String],
                     variableTypes: Array[Int],
                     variableAssignPlaceholders: Array[String],
                     assignVariableOp: String,
                     extraVariables: Array[String],
                     extraVariableTypes: Array[Int],
                     extraVariableAssignPlaceholders: Array[String],
                     assignExtraVariableOp: String,
                     gradVariables: Array[String],
                     restoreOp: String,
                     restorePathPlaceholder: String,
                     saveOp: String,
                     savePathPlaceholder: String,
                     updateOp: String,
                     trainOp: Option[String],
                     initOp: Option[String],
                     defaultTensorValue: Array[Array[Float]],
                     metricsNames: Array[String])

/**
 * TFSubGraph will only be used in DistriOptimizer for the purpose of training a TensorFlow
 * model using multiple optimization methods based on variable names.
 * Applying a TFTrainingHelper2 layer by name will get a corresponding instance of TFSubGraph.
 *
 * In DistriOptimizer.optimize(), TFSubGraph will only be used to get the sizes and offsets of
 * each weight portion, slice on the original weights and gradients and apply the optimization
 * method accordingly.
 * The gradients of TFSubGraph will never be used and thus a dummy Tensor is put as a placeholder.
 */
private[zoo] class TFSubGraph(
        weights: Array[Tensor[Float]]) extends AbstractModule[Activity, Activity, Float] {
  override def updateOutput(input: Activity): Activity = {
    input
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput
  }

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    (weights, weights.map(_ => Tensor[Float]()))
  }
}

case class SessionConfig(intraOpParallelismThreads: Int = 1,
                         interOpParallelismThreads: Int = 1,
                         usePerSessionThreads: Boolean = true) {

  // Ideally we should use the following code, however, importing tensorflow proto
  // will conflict with bigdl.

  //  val defaultSessionConfig = ConfigProto.newBuilder()
  //    .setInterOpParallelismThreads(1)
  //    .setIntraOpParallelismThreads(1)
  //    .setUsePerSessionThreads(true)
  //    .build().toByteArray

  def toByteArray(): Array[Byte] = {
    val intraSeq = if (intraOpParallelismThreads > 0) {
      Seq(16, intraOpParallelismThreads)
    } else {
      Seq[Int]()
    }
    val interSeq = if (interOpParallelismThreads > 0) {
      Seq(40, interOpParallelismThreads)
    } else {
      Seq[Int]()
    }
    val perSessSeq = if (usePerSessionThreads) {
      Seq(72, 1)
    } else {
      Seq[Int]()
    }

    (intraSeq ++ interSeq ++ perSessSeq).map(_.toByte).toArray
  }
}
