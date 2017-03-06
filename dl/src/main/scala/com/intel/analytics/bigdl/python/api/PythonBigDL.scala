/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.python.api

import java.util.{ArrayList => JArrayList, HashMap => JHashMap, List => JList, Map => JMap}

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{Sample => JSample, _}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.numeric._
import com.intel.analytics.bigdl.optim.{Optimizer, _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, Table}
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.rdd.RDD
import scala.collection.Map

import scala.collection.JavaConverters._
import scala.language.existentials
import scala.reflect.ClassTag

case class Sample(features: JList[Any],
                  label: JList[Any],
                  featuresShape: JList[Int],
                  labelShape: JList[Int],
                  bigdlType: String)
case class TestResult(val result: Float, count: Int, val method: String)


object PythonBigDL{
  val floatInstance = new PythonBigDL[Float]()

  val doubleInstance = new PythonBigDL[Double]()

  def ofFloat(): PythonBigDL[Float] = floatInstance

  def ofDouble(): PythonBigDL[Double] = doubleInstance

  def getInitMethod(initMethod: String): InitializationMethod = {
    initMethod.toLowerCase() match {
      case "xavier" => Xavier
      case "default" => Default
      case "bilinearfiller" => BilinearFiller
      case m: String => throw new IllegalArgumentException(s"Not supported init method: ${m}")
    }
  }
}

class PythonBigDL[T: ClassTag](implicit ev: TensorNumeric[T]) extends Serializable {

  private val typeName = {
    val cls = implicitly[ClassTag[T]].runtimeClass
    cls.getSimpleName
  }

  private def toValidationMethod(vMethods: JList[String]): Array[ValidationMethod[T]] = {
    vMethods.toArray.map{case m: String => m.toLowerCase()}.map {
      case "top1" => new Top1Accuracy[T]()
      case "top5" => new Top5Accuracy[T]()
      case "loss" => new Loss[T]()
      case m: String => throw new RuntimeException(s"not supported validation method: $m")
    }
  }

  private def validationMethodToStr(method: ValidationMethod[T]): String = {
    method match {
      case _: Top1Accuracy[T] => "top1"
      case _: Top5Accuracy[T] => "top5"
      case _: Loss[T] => "loss"
      case _ => throw new RuntimeException(s"not supported validation method: $method")
    }
  }

  def toPySample(sample: JSample[T]): Sample = {
    val featureList = sample.feature().contiguous().storage().toArray[T].toList.asJava
    val labelList = sample.label().contiguous().storage().toArray[T].toList.asJava
    val cls = implicitly[ClassTag[T]].runtimeClass
    Sample(featureList.asInstanceOf[JList[Any]],
      labelList.asInstanceOf[JList[Any]],
      sample.feature().size().toList.asJava,
      sample.label().size().toList.asJava,
      cls.getSimpleName)
  }

  def toSample(record: Sample): JSample[T] = {
    require(record.bigdlType == this.typeName,
      s"record.bigdlType: ${record.bigdlType} == this.typeName: ${this.typeName}")
    val sample = this.typeName match {
      case "float" =>
        JSample[Float]().set(
          record.features.asInstanceOf[JList[Double]].asScala.map(_.toFloat).toArray[Float],
          (record.label.asInstanceOf[JList[Double]]).asScala.map(_.toFloat).toArray[Float],
          record.featuresShape.asScala.toArray[Int],
          record.labelShape.asScala.toArray[Int])
      case "double" =>
        JSample[Double]().set(
          record.features.asInstanceOf[JList[Double]].asScala.toArray[Double],
          (record.label.asInstanceOf[JList[Double]]).asScala.toArray[Double],
          record.featuresShape.asScala.toArray[Int],
          record.labelShape.asScala.toArray[Int])
      case t: String =>
        throw new IllegalArgumentException(s"Not supported type: ${t}")
    }
    sample.asInstanceOf[JSample[T]]
  }
  private def batching(rdd: RDD[Sample], batchSize: Int)
  : DistributedDataSet[MiniBatch[T]] = {
    val recordRDD = rdd.map(toSample(_))
    (DataSet.rdd(recordRDD) -> new SampleToBatch[T](batchSize))
      .asInstanceOf[DistributedDataSet[MiniBatch[T]]]
  }

  def createSequential(): Sequential[T] = {
    Sequential[T]()
  }

  def createLinear(inputSize: Int, outputSize: Int, initMethod: String): Linear[T] = {
    Linear[T](inputSize, outputSize, PythonBigDL.getInitMethod(initMethod))
  }

  def createReLU(): ReLU[T] = {
    ReLU[T]()
  }

  def createTanh(): Tanh[T] = {
    Tanh[T]()
  }

  def createEcho(): Echo[T] = {
    Echo[T]()
  }

  def createLogSoftMax(): LogSoftMax[T] = {
    LogSoftMax[T]()
  }

  def createSpatialMaxPooling(kW: Int,
                              kH: Int,
                              dW: Int,
                              dH: Int,
                              padW: Int = 0,
                              padH: Int = 0): SpatialMaxPooling[T] = {
    SpatialMaxPooling[T](kW,
      kH,
      dW,
      dH,
      padW,
      padH)
  }

  def createSpatialConvolution(nInputPlane: Int,
                               nOutputPlane: Int,
                               kernelW: Int,
                               kernelH: Int,
                               strideW: Int = 1,
                               strideH: Int = 1,
                               padW: Int = 0,
                               padH: Int = 0,
                               nGroup: Int = 1,
                               propagateBack: Boolean = true,
                               initMethod: String = "default")
  : SpatialConvolution[T] = {
    SpatialConvolution[T](nInputPlane,
      nOutputPlane,
      kernelW,
      kernelH,
      strideW,
      strideH,
      padW,
      padH,
      nGroup,
      propagateBack,
      PythonBigDL.getInitMethod(initMethod))
  }

  def createReshape(size: JList[Int]): Reshape[T] = {
    Reshape(size.asScala.toArray)
  }

  //   Optimizer
  def createClassNLLCriterion: ClassNLLCriterion[T] = {
    ClassNLLCriterion[T]()
  }

  def createMSECriterion: MSECriterion[T] = {
    MSECriterion[T]()
  }

  def createValidator(model: AbstractModule[Activity, Activity, T],
                      valRDD: JavaRDD[Sample],
                      batchSize: Int)
  : Validator[T, MiniBatch[T]] = {
    Validator(model, batching(valRDD, batchSize))
  }

  def modelTest(model: AbstractModule[Activity, Activity, T],
                valRDD: JavaRDD[Sample],
                batchSize: Int,
                valMethods: JList[String])
  : JList[TestResult] = {
    val validator = Validator(model, batching(valRDD, batchSize))
    val resultArray = validator.test(toValidationMethod(valMethods))
    val testResultArray = resultArray.map{result =>
      TestResult(result._1.result()._1, result._1.result()._2,
        validationMethodToStr(result._2))
    }
    testResultArray.toList.asJava
  }

  def modelFromPath(path: String): AbstractModule[Activity, Activity, T] = {
    Module.load[T](path)
  }

  def modelPredictRDD(model: AbstractModule[Activity, Activity, T],
                      dataRdd: JavaRDD[Sample]): JavaRDD[Sample] = {
    val result = predict(model, dataRdd.rdd.map(toSample(_)))
    result.map(toPySample(_))

  }

  def modelGetParameters(model: AbstractModule[Activity, Activity, T])
  : JMap[Any, JMap[Any, JList[JList[Any]]]] = {
    model.getParametersTable().getState().mapValues {
      case name2Values: Table =>
        name2Values.getState().mapValues {
          case t : Tensor[T] =>
            val tensorClone = t.clone()
            val item = List(tensorClone.storage().toList.asJava.asInstanceOf[JList[Any]],
              tensorClone.size().toList.asJava.asInstanceOf[JList[Any]]).asJava
            item
        }.asJava
    }.asJava
  }

  def predict(model: AbstractModule[Activity, Activity, T],
                      dataRdd: RDD[JSample[T]]): RDD[JSample[T]] = {
    val modelBroadCast = dataRdd.sparkContext.broadcast(model.evaluate())
    dataRdd.mapPartitions {partition =>
      val localModel = modelBroadCast.value.cloneModule()
      partition.map {sample =>
        val output = localModel.forward(sample.feature()).toTensor[T]
        JSample(sample.feature(), output)
      }
    }
  }

  def createMaxEpoch(max: Int): Trigger = {
    Trigger.maxEpoch(max)
  }

  def createEveryEpoch(): Trigger = {
    Trigger.everyEpoch
  }

  def createSeveralIteration(interval: Int): Trigger = {
    Trigger.severalIteration(interval)
  }

  def createMaxIteration(max: Int): Trigger = {
    Trigger.maxIteration(max)
  }

  def createOptimizer(model: AbstractModule[Activity, Activity, T],
                      trainingRdd: JavaRDD[Sample],
                      criterion: Criterion[T],
                      optimMethod: String,
                      state: JMap[Any, Any],
                      endTrigger: Trigger,
                      batchSize: Int): Optimizer[T, MiniBatch[T]] = {
    val optimizer = new DistriOptimizer(
      model = model,
      dataset = batching(trainingRdd, batchSize),
      criterion = criterion
    ).asInstanceOf[Optimizer[T, MiniBatch[T]]]
    // TODO: we should provide a more convenient way to create Table
    val stateTable = new Table()
    state.asScala.foreach{case (key, value) =>
      stateTable.update(key, value)
    }
    optimizer.setState(stateTable)

    optimizer.setEndWhen(endTrigger)

    optimMethod.toLowerCase() match {
      case "sgd" =>
        optimizer.setOptimMethod(new SGD())
      case "adagrad" =>
        optimizer.setOptimMethod(new Adagrad())
      case "lbfgs" =>
        optimizer.setOptimMethod(new LBFGS())
      case n: String => throw new IllegalArgumentException(s"Not supported type: $n")
    }
    // TODO: remove this
    optimizer.disableCheckSingleton()

    optimizer
  }

  def setValidation(optimizer: Optimizer[T, MiniBatch[T]],
                    batchSize: Int,
                    trigger: Trigger,
                    valRdd: JavaRDD[Sample],
                    vMethods: JList[String]): Unit = {
    optimizer.setValidation(trigger, batching(valRdd, batchSize.toInt),
      toValidationMethod(vMethods))
  }

  def setCheckPoint(optimizer: Optimizer[T, MiniBatch[T]],
                    trigger: Trigger,
                    checkPointPath: String,
                    isOverwrite: Boolean): Unit = {
    optimizer.setCheckpoint(checkPointPath, trigger)
    if(isOverwrite) {
      optimizer.overWriteCheckpoint()
    }
  }

  def initEngine(nodeNum: Int, coreNum: Int): Unit = {
    Engine.setNodeAndCore(nodeNum, coreNum)
  }
}



