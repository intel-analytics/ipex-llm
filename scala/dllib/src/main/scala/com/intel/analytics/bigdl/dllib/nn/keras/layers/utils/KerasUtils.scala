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

package com.intel.analytics.zoo.pipeline.api.keras.layers.utils

import com.intel.analytics.bigdl.Criterion
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.keras.{KerasIdentityWrapper, KerasLayer, KerasLayerWrapper, Sequential => KSequential, SoftMax => KSoftMax}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat, TensorModule}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{MultiShape, Shape, SingleShape}
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.metrics.{AUC, Accuracy, BinaryAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy, Top5Accuracy => ZooTop5Accuracy}
import com.intel.analytics.zoo.pipeline.api.keras.models.KerasNet
import com.intel.analytics.zoo.pipeline.api.keras.objectives._
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object KerasUtils {

  def getPadsFromBorderMode(borderMode: String = "valid",
      paddings: Array[Int] = null): (Int, Int) = {
    if (paddings != null && !paddings.isEmpty) {
      require(paddings.length == 2)
      (paddings(0), paddings(1))
    } else if (borderMode == "same") {
      // padH, padW
      (-1, -1)
    } else {
      (0, 0)
    }
  }

  def getInitMethod(init: String, limits: Array[Double] = null): InitializationMethod = {
    init.toLowerCase() match {
      case "glorot_uniform" => Xavier
      case "one" => Ones
      case "zero" => Zeros
      case "uniform" =>
        if (limits == null) {
          RandomUniform(-0.05, 0.05)
        }
        else {
          RandomUniform(limits.head, limits(1))
        }
      case "normal" =>
        if (limits == null) {
          RandomNormal(0.0, 0.05)
        } else {
          RandomNormal(limits.head, limits(1))
        }
      case _ => throw new IllegalArgumentException(s"Unsupported initialization method: " +
        s"${init.toLowerCase()}")
    }
  }

  def getKerasActivation[T : ClassTag] (activation: String)
    (implicit ev: TensorNumeric[T]): KerasLayer[Tensor[T], Tensor[T], T] = {
    if (activation == null) { return null }
    if (activation.toLowerCase() == "softmax") {
      com.intel.analytics.zoo.pipeline.api.keras.layers.SoftMax[T]()
    } else {
      val torchActivation = getTorchActivation(activation)
      new KerasIdentityWrapper[T](torchActivation)
        .asInstanceOf[KerasLayer[Tensor[T], Tensor[T], T]]
    }
  }

  def getActivationName[T: ClassTag](activation: AbstractModule[_, _, T]): String = {
    if (activation == null) {
      throw new IllegalArgumentException("activation is null")
    } else {
      activation match {
        case _: Tanh[T] => "tanh"
        case _: Sigmoid[T] => "sigmoid"
        case _: ReLU[T] => "relu"
        case _: com.intel.analytics.bigdl.nn.SoftMax[T] => "softmax"
        case _: SoftPlus[T] => "softplus"
        case _: SoftSign[T] => "softsign"
        case _: HardSigmoid[T] => "hard_sigmoid"
        case _: ReLU6[T] => "relu6"
        case _: TanhShrink[T] => "tanh_shrink"
        case _: SoftMin[T] => "softmin"
        case _: LogSigmoid[T] => "log_sigmoid"
        case _: LogSoftMax[T] => "log_softmax"
        case _: Identity[T] => "linear"
        case _: com.intel.analytics.zoo.pipeline.api.keras.layers.SoftMax[T] => "softmax"
        case _ => throw new IllegalArgumentException("unkown activation"
          + activation.getClass.getName)
      }
    }

  }

  def getTorchActivation[T : ClassTag] (activation: String)
    (implicit ev: TensorNumeric[T]): AbstractModule[Tensor[T], Tensor[T], T] = {
    if (activation == null) null
    else {
      activation.toLowerCase() match {
          case "tanh" => Tanh[T]()
          case "sigmoid" => Sigmoid[T]()
          case "relu" => ReLU[T]()
          case "softmax" =>
                com.intel.analytics.bigdl.nn.SoftMax[T]()
          case "softplus" => SoftPlus[T]()
          case "softsign" => SoftSign[T]()
          case "hard_sigmoid" => HardSigmoid[T]()
          case "relu6" => ReLU6[T]()
          case "tanh_shrink" => TanhShrink[T]()
          case "softmin" => SoftMin[T]()
          case "log_sigmoid" => LogSigmoid[T]()
          case "log_softmax" => LogSoftMax[T]()
          case "linear" => Identity[T]().asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
          case _ => throw new IllegalArgumentException(s"Invalid activation: " +
            s"${activation.toLowerCase}. Only simple activations can be constructed using string")
      }
    }
  }

  def computeConvOutputLength(
    inputLength: Int,
    filterSize: Int,
    borderMode: String,
    stride: Int,
    dilation: Int = 1): Int = {
    val dilatedFilterSize = filterSize + (filterSize - 1) * (dilation - 1)
    val outputLength = borderMode match {
      case "valid" => inputLength - dilatedFilterSize + 1
      case "same" => inputLength
    }
    (outputLength + stride - 1) / stride
  }

  def getPadsFromBorderMode3D(
    borderMode: String = "valid"): (Int, Int, Int) = {
    if (borderMode == "same") {
      // padT, padH, padW
      (-1, -1, -1)
    } else {
      (0, 0, 0)
    }
  }

  def toBigDLFormat(dimOrdering: String): DataFormat = {
    require(dimOrdering.toLowerCase() == "tf" || dimOrdering.toLowerCase() == "th",
      s"Dim ordering must be either tf or th, but got ${dimOrdering.toLowerCase()}")
    dimOrdering.toLowerCase() match {
      case "tf" => DataFormat.NHWC
      case "th" => DataFormat.NCHW
    }
  }

  def toBigDLFormat5D(dimOrdering: String): String = {
    require(dimOrdering.toLowerCase() == "tf" || dimOrdering.toLowerCase() == "th",
      s"Dim ordering must be either tf or th, but got ${dimOrdering.toLowerCase()}")
    dimOrdering.toLowerCase() match {
      case "tf" => "CHANNEL_LAST"
      case "th" => "CHANNEL_FIRST"
    }
  }

  def toBigDLCriterion[T : ClassTag](loss: String)
    (implicit ev: TensorNumeric[T]): Criterion[T] = {
    loss.toLowerCase() match {
      case "binary_crossentropy" => BinaryCrossEntropy[T]()
      case "categorical_crossentropy" =>
        com.intel.analytics.zoo.pipeline.api.keras.objectives.CategoricalCrossEntropy[T]()
      case "mse" => MeanSquaredError[T]()
      case "mean_squared_error" => MeanSquaredError[T]()
      case "mae" => MeanAbsoluteError[T]()
      case "mean_absolute_error" => MeanAbsoluteError[T]()
      case "hinge" => Hinge[T]()
      case "mape" => MeanAbsolutePercentageError[T]()
      case "mean_absolute_percentage_error" => MeanAbsolutePercentageError[T]()
      case "msle" => MeanSquaredLogarithmicError[T]()
      case "mean_squared_logarithmic_error" => MeanSquaredLogarithmicError[T]()
      case "squared_hinge" => SquaredHinge[T]()
      case "sparse_categorical_crossentropy" => SparseCategoricalCrossEntropy[T]()
      case "kld" => KullbackLeiblerDivergence[T]()
      case "kullback_leibler_divergence" => KullbackLeiblerDivergence[T]()
      case "cosine_proximity" => CosineProximity[T]()
      case "poisson" => Poisson[T]()
      case "rank_hinge" => RankHinge[T]()
      case _ => throw new IllegalArgumentException(s"Unsupported loss: $loss")
    }
  }

  def toBigDLOptimMethod[T: ClassTag](optimMethod: String)
    (implicit ev: TensorNumeric[T]): OptimMethod[T] = {
    optimMethod.toLowerCase() match {
      case "sgd" => new SGD[T](learningRate = 0.01)
      case "rmsprop" => new RMSprop[T](learningRate = 0.001, decayRate = 0.9)
      case "adamax" => new Adamax[T](Epsilon = 1e-8)
      case "adagrad" => new Adagrad[T](learningRate = 0.01)
      case "adadelta" => new Adadelta[T](decayRate = 0.95, Epsilon = 1e-8)
      case "adam" => new Adam[T]()
    }
  }

  private def mappingForAcc[T: ClassTag](loss: String)(implicit ev: TensorNumeric[T])
  : ValidationMethod[T] = {
    loss.toLowerCase() match {
      case "sparse_categorical_crossentropy" => new SparseCategoricalAccuracy[T]()
      case "categorical_crossentropy" => new CategoricalAccuracy[T]()
      case "binary_crossentropy" => new BinaryAccuracy[T]()
      case _ => throw new IllegalArgumentException(
        s"Unsupported metric: accuracy and loss: ${loss} combination")
    }
  }

  def toBigDLMetrics[T: ClassTag](metrics: List[String], loss: String)
    (implicit ev: TensorNumeric[T]): List[ValidationMethod[T]] = {
    if (metrics == null) {
      null
    } else {
      metrics.map { metric =>
        metric.toLowerCase() match {
          case "accuracy" => mappingForAcc(loss)
          case "acc" => mappingForAcc(loss)
          case "top5accuracy" => new ZooTop5Accuracy[T]()
          case "top5acc" => new ZooTop5Accuracy[T]()
          case "mae" => new MAE[T]()
          case "auc" => new AUC[T]()
          case "loss" => new Loss[T]()
          case "treennaccuracy" => new TreeNNAccuracy[T]()
          case _ => throw new IllegalArgumentException(s"Unsupported metric: $metric")
        }
      }
    }
  }

  def addBatch(shape: Shape): Shape = {
    if (shape == null) {
      return null
    }
    if (shape.isInstanceOf[SingleShape]) {
      Shape((List(-1) ++ shape.toSingle()).toArray)
    } else {
      MultiShape(shape.toMulti().map {addBatch})
    }
  }

  def removeBatch(shape: Shape): Shape = {
    if (shape == null) {
      return null
    }
    if (shape.isInstanceOf[SingleShape]) {
      Shape(shape.toSingle().slice(1, shape.toSingle().length).toArray)
    } else {
      MultiShape(shape.toMulti().map {removeBatch})
    }
  }

  def fuse[T: ClassTag](
      torchLayer: AbstractModule[Activity, Activity, T],
      kerasActivation: KerasLayer[Tensor[T], Tensor[T], T],
      batchInputShape: Shape)
      (implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {
    if (kerasActivation == null) {
      torchLayer
    } else {
      val wrapper = KSequential[T]()
      wrapper.add(new KerasLayerWrapper[T](torchLayer,
        removeBatch(batchInputShape)))
      wrapper.add(kerasActivation)
      wrapper.setName(torchLayer.getName())
      wrapper.build(batchInputShape)
      wrapper
    }
  }

  private[zoo] def invokeMethod(obj: Object, methodName: String, args: Object*): Object = {
    val clazz = obj.getClass()
    val method =
      try {
      clazz.getMethod(methodName, args.map(_.getClass): _*)
    } catch {
        case t: Throwable =>
          val methods = clazz.getMethods().filter(_.getName() == methodName)
          require(methods.length == 1,
            s"We should only found one result, but got ${methodName}: ${methods.length}")
          methods(0)
    }
    method.invoke(obj, args: _*)
  }

  private[zoo] def invokeMethodWithEv[T: ClassTag](
        obj: String,
        methodName: String,
        args: Object*)(implicit ev: TensorNumeric[T]): Object = {
    val clazz = Class.forName(obj)
    val method =
      try {
        clazz.getMethod(methodName, args.map(_.getClass): _*)
      } catch {
        case t: Throwable =>
          val methods = clazz.getMethods().filter(_.getName() == methodName)
          require(methods.length == 1,
            s"We should only found one result, but got ${methodName}: ${methods.length}")
          methods(0)
      }
    val argsWithTag = args ++ Seq(implicitly[reflect.ClassTag[T]], ev)
    method.invoke(obj, argsWithTag: _*)
  }

  private[zoo] def invokeMethodWithEv[T: ClassTag](
        obj: Object,
        methodName: String,
        args: Object*)(implicit ev: TensorNumeric[T]): Object = {
    val argsWithTag = args ++ Seq(implicitly[reflect.ClassTag[T]], ev)
    invokeMethod(obj, methodName, argsWithTag: _*)
  }

  /**
   * Count the total number of parameters for a KerasLayer.
   * Return a tuple (total params #, trainable params #)
   */
  def countParams[T: ClassTag](layer: KerasLayer[Activity, Activity, T]): (Int, Int) = {
    val (weights, gradWeights) = layer.parameters()
    var count = 0
    for (w <- weights) {
      count += w.nElement()
    }
    if (layer.isInstanceOf[KerasNet[T]]) {
      val modules = layer.labor.asInstanceOf[Container[Activity, Activity, T]].modules
      var trainable = 0
      for (module <- modules) {
        trainable += countParams[T](module.asInstanceOf[KerasLayer[Activity, Activity, T]])._2
      }
      (count, trainable)
    }
    else {
      if (layer.asInstanceOf[Net].isFrozen()) {
        (count, 0)
      }
      else {
        (count, count)
      }
    }
  }

  /**
   * Return the layer summary information as an array of String, in the order of:
   * Layer (type), OutputShape, Param #
   */
  def getLayerSummary[T: ClassTag](layer: KerasLayer[Activity, Activity, T]): Array[String] = {
    val outputShape = strShape(layer.getOutputShape())
    val name = layer.getName
    val className = layer.getClass.getSimpleName
    Array(name + " (" + className + ")", outputShape.toString,
      KerasUtils.countParams(layer)._1.toString)
  }

  /**
   * Together with the layer summary of a node, also return the name of the node(s)
   * that it is connected to.
   * If there are multiple connected nodes, they will be combined by ", "
   */
  def getNodeSummary[T: ClassTag](node: ModuleNode[T]): Array[String] = {
    val layer = node.element.asInstanceOf[KerasLayer[Activity, Activity, T]]
    val results = getLayerSummary(layer)
    var connectedTo = ""
    val prevNodes = node.prevNodes
    for (i <- prevNodes.indices) {
      if (i > 0) connectedTo += ", "
      connectedTo += prevNodes(i).element.getName
    }
    results ++ Array(connectedTo)
  }

  /**
   * Print the summary of a node in a line.
   * Return a tuple (total params #, trainable params #) of this node.
   */
  def printNodeSummary[T: ClassTag](
      node: ModuleNode[T],
      lineLength: Int = 120,
      positions: Array[Double] = Array(.33, .55, .67, 1)): (Int, Int) = {
    printRow(getNodeSummary(node), lineLength, positions)
    countParams(node.element.asInstanceOf[KerasLayer[Activity, Activity, T]])
  }

  /**
   * Print a row containing several fields.
   *
   * @param fields The fields to be printed out.
   * @param lineLength The total length of a printed line.
   * @param positions The maximum absolute length proportion(%) of each field.
   *                  Default is Array(.33, .55, .67, 1), meaning that
   *                  the first field will occupy up to 33% of lineLength,
   *                  the second field will occupy up to (55-33)% of lineLength,
   *                  the third field will occupy up to (67-55)% of lineLength,
   *                  the fourth field will occupy the remaining line (100-67)%.
   *                  If the field has a larger length, the remaining part will be trimmed.
   *                  If the field has a smaller length, the remaining part will be white spaces.
   * @param includeSplitLine Whether to add a split line after printing one row.
   * @param splitChar The character to compose the split line.
   */
  def printRow(
      fields: Array[String],
      lineLength: Int = 120,
      positions: Array[Double] = Array(.33, .55, .67, 1),
      includeSplitLine: Boolean = true,
      splitChar: Char = '_'): Unit = {
    val fieldLengths = ArrayBuffer[Int]()
    for (i <- positions.indices) {
      if (i > 0) {
        val len = (positions(i) - positions(i-1)) * lineLength
        require(len > 0, s"Invalid positions specified: ${positions(i)} < ${positions(i-1)}")
        fieldLengths.append(len.toInt)
      }
      else fieldLengths.append((positions(i)*lineLength).toInt)
    }
    var line = ""
    // If there are multiple connected to nodes, each will be printed in a separate line.
    var nodes = Array[String]()
    for (i <- fields.indices) {
      if (i > 0) line += " "
      if (i == 3) {
        nodes = fields(i).split(", ")
        line += nodes(0)
      }
      else {
        line += fields(i)
      }
      val maxLength = fieldLengths.take(i + 1).sum
      if (line.length > maxLength) {
        line = line.substring(0, maxLength)
      }
      else {
        line += " " * (maxLength - line.length)
      }

    }
    println(line)
    // If there are multiple connected to nodes, print the remaining each in a separate line
    // without the split line.
    for (node <- nodes.slice(1, nodes.length)) {
      printRow(Array("", "", "", node), lineLength, positions, includeSplitLine = false)
    }
    if (includeSplitLine) printSplitLine(splitChar, lineLength)
  }

  /**
   * Print a split line that repeats the 'char' for 'lineLength' times.
   */
  def printSplitLine(char: Char, lineLength: Int = 120): Unit = {
    val str = char.toString
    println(str * lineLength)
  }

  /**
   * Convert a Shape to String format using 'None' to indicate batch,
   * which is the same as Keras. Used to print out the shape.
   *
   * For example,
   * (None, 10) will be returned for Shape(-1, 10), a SingleShape.
   * (None, 10) (None, 8) will be returned for a MultiShape which consists of
   * Shape(-1, 10), Shape(-1, 8).
   */
  def strShape(shape: Shape): String = {
    shape match {
      case s: SingleShape =>
        val res = "(" + s.toSingle().mkString(", ") + ")"
        res.replaceFirst("-1", "None")
      case m: MultiShape =>
        val shapes = m.toMulti()
        var res = ""
        for (shape <- shapes) res = res + strShape(shape) + " "
        res
    }
  }

  /**
   * classes: RDD of 1-based label.
   * If zeroBasedLabel is true, convert to RDD of 0-based label.
   * Otherwise, just return classes itself.
   */
  def toZeroBasedLabel(
      zeroBasedLabel: Boolean = true,
      classes: RDD[Int]): RDD[Int] = {
    if (zeroBasedLabel) {
      classes.map(_ - 1)
    }
    else {
      classes
    }
  }

  def validateBatchSize(batchSize: Int): Unit = {
    val totalCores = EngineRef.getCoreNumber() * EngineRef.getNodeNumber()
    require(batchSize % totalCores == 0,
      s"BatchSize: ${batchSize} cannot be divided by ${totalCores}")
  }

  def tril[T: ClassTag](x: Tensor[T])(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(x.dim() == 2, "tril expects a matrix!")
    val stride1 = x.stride(1)
    val stride2 = x.stride(2)
    for (i <- 0 until x.size(1)) {
      for (c <- i + 1 until x.size(2)) {
        val data = x.storage().array
        data(i * stride1 + c * stride2) = ev.fromType(0)
      }
    }
    x
  }
}
