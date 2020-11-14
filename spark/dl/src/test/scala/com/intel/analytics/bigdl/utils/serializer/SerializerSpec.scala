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
package com.intel.analytics.bigdl.utils.serializer

import java.io.File
import java.lang.reflect.Modifier

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Tensor, TensorNumericMath}
import com.intel.analytics.bigdl.utils.BigDLSpecHelper
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import org.reflections.Reflections
import org.reflections.scanners.SubTypesScanner
import org.reflections.util.{ClasspathHelper, ConfigurationBuilder, FilterBuilder}

import collection.JavaConverters._
import scala.collection.mutable
import scala.reflect.ClassTag
import scala.reflect.runtime.universe

class SerializerSpec extends BigDLSpecHelper {

  private val excluded = Set[String](
    "com.intel.analytics.bigdl.nn.CellUnit",
    "com.intel.analytics.bigdl.nn.tf.ControlDependency",
    "com.intel.analytics.bigdl.utils.tf.AdapterForTest",
    "com.intel.analytics.bigdl.utils.serializer.TestModule",
    "com.intel.analytics.bigdl.utils.TestModule",
    "com.intel.analytics.bigdl.utils.ExceptionTest",
    "com.intel.analytics.bigdl.utils.serializer.SubModuleOne",
    "com.intel.analytics.bigdl.utils.serializer.SubModuleTwo",
    "com.intel.analytics.bigdl.nn.mkldnn.AvgPooling",
    "com.intel.analytics.bigdl.nn.mkldnn.CAddTable",
    "com.intel.analytics.bigdl.nn.mkldnn.ConcatTable",
    "com.intel.analytics.bigdl.nn.mkldnn.DnnBase",
    "com.intel.analytics.bigdl.nn.mkldnn.Identity",
    "com.intel.analytics.bigdl.nn.mkldnn.Input",
    "com.intel.analytics.bigdl.nn.mkldnn.JoinTable",
    "com.intel.analytics.bigdl.nn.mkldnn.Linear",
    "com.intel.analytics.bigdl.nn.mkldnn.LRN",
    "com.intel.analytics.bigdl.nn.mkldnn.MaxPooling",
    "com.intel.analytics.bigdl.nn.mkldnn.ReLU",
    "com.intel.analytics.bigdl.nn.mkldnn.ReorderMemory",
    "com.intel.analytics.bigdl.nn.mkldnn.SelectTable",
    "com.intel.analytics.bigdl.nn.mkldnn.Sequential",
    "com.intel.analytics.bigdl.nn.mkldnn.SoftMax",
    "com.intel.analytics.bigdl.nn.mkldnn.SpatialBatchNormalization",
    "com.intel.analytics.bigdl.nn.mkldnn.SpatialConvolution",
    "com.intel.analytics.bigdl.nn.mkldnn.Dropout",
    "com.intel.analytics.bigdl.nn.mkldnn.DnnGraph",
    "com.intel.analytics.bigdl.nn.mkldnn.BlasWrapper",
    "com.intel.analytics.bigdl.nn.mkldnn.Output",
    "com.intel.analytics.bigdl.nn.mkldnn.InputWrapper",
    "com.intel.analytics.bigdl.utils.intermediate.IRGraph",
    "com.intel.analytics.bigdl.nn.mkldnn.RNN"
  )

  // Maybe one serial test class contains multiple module test
  // Also keras layer main/test class mapping are weired
  private val unRegularNameMapping = Map[String, String](
    // Many to one mapping
    "com.intel.analytics.bigdl.nn.ops.Enter" ->
      "com.intel.analytics.bigdl.nn.ops.ControlOpsSerialTest",
    "com.intel.analytics.bigdl.nn.tf.Enter" ->
      "com.intel.analytics.bigdl.nn.tf.ControlOpsSerialTest",
    "com.intel.analytics.bigdl.nn.ops.NextIteration" ->
      "com.intel.analytics.bigdl.nn.ops.ControlOpsSerialTest",
    "com.intel.analytics.bigdl.nn.tf.NextIteration" ->
      "com.intel.analytics.bigdl.nn.tf.ControlOpsSerialTest",
    "com.intel.analytics.bigdl.nn.ops.Exit" ->
      "com.intel.analytics.bigdl.nn.ops.ControlOpsSerialTest",
    "com.intel.analytics.bigdl.nn.tf.Exit" ->
      "com.intel.analytics.bigdl.nn.tf.ControlOpsSerialTest",
    "com.intel.analytics.bigdl.nn.ops.LoopCondition" ->
      "com.intel.analytics.bigdl.nn.ops.ControlOpsSerialTest",
    "com.intel.analytics.bigdl.nn.tf.LoopCondition" ->
      "com.intel.analytics.bigdl.nn.tf.ControlOpsSerialTest",
    "com.intel.analytics.bigdl.nn.ops.StackCreator" ->
      "com.intel.analytics.bigdl.nn.ops.StackOpsSerialTest",
    "com.intel.analytics.bigdl.nn.tf.StackCreator" ->
      "com.intel.analytics.bigdl.nn.tf.StackOpsSerialTest",
    "com.intel.analytics.bigdl.nn.ops.StackPush" ->
      "com.intel.analytics.bigdl.nn.ops.StackOpsSerialTest",
    "com.intel.analytics.bigdl.nn.tf.StackPush" ->
      "com.intel.analytics.bigdl.nn.tf.StackOpsSerialTest",
    "com.intel.analytics.bigdl.nn.ops.StackPop" ->
      "com.intel.analytics.bigdl.nn.ops.StackOpsSerialTest",
    "com.intel.analytics.bigdl.nn.tf.StackPop" ->
      "com.intel.analytics.bigdl.nn.tf.StackOpsSerialTest",
    "com.intel.analytics.bigdl.nn.ops.TensorArrayWrite" ->
      "com.intel.analytics.bigdl.nn.ops.TensorArraySerialTest",
    "com.intel.analytics.bigdl.nn.tf.TensorArrayWrite" ->
      "com.intel.analytics.bigdl.nn.ops.TensorArraySerialTest",
    "com.intel.analytics.bigdl.nn.ops.TensorArrayRead" ->
      "com.intel.analytics.bigdl.nn.ops.TensorArraySerialTest",
    "com.intel.analytics.bigdl.nn.tf.TensorArrayRead" ->
      "com.intel.analytics.bigdl.nn.ops.TensorArraySerialTest",
    "com.intel.analytics.bigdl.nn.ops.TensorArrayGrad" ->
      "com.intel.analytics.bigdl.nn.ops.TensorArraySerialTest",
    "com.intel.analytics.bigdl.nn.tf.TensorArrayGrad" ->
      "com.intel.analytics.bigdl.nn.ops.TensorArraySerialTest",
    "com.intel.analytics.bigdl.nn.tf.TensorArrayCreator" ->
      "com.intel.analytics.bigdl.nn.ops.TensorArrayScatterSerialTest",
    "com.intel.analytics.bigdl.nn.tf.TensorArrayScatter" ->
      "com.intel.analytics.bigdl.nn.ops.TensorArrayScatterSerialTest",
    "com.intel.analytics.bigdl.nn.tf.TensorArrayGather" ->
      "com.intel.analytics.bigdl.nn.ops.TensorArrayScatterSerialTest",
    "com.intel.analytics.bigdl.nn.tf.TensorArrayClose" ->
      "com.intel.analytics.bigdl.nn.ops.TensorArrayScatterSerialTest",
    "com.intel.analytics.bigdl.nn.tf.TensorArrayConcat" ->
      "com.intel.analytics.bigdl.nn.ops.TensorArraySplitSerialTest",
    "com.intel.analytics.bigdl.nn.tf.TensorArraySplit" ->
      "com.intel.analytics.bigdl.nn.ops.TensorArraySplitSerialTest",
    "com.intel.analytics.bigdl.nn.tf.TensorArraySize" ->
      "com.intel.analytics.bigdl.nn.ops.TensorArraySplitSerialTest",


    // Keras layers
    "com.intel.analytics.bigdl.nn.keras.Input" ->
      "com.intel.analytics.bigdl.keras.nn.InputSerialTest",
    "com.intel.analytics.bigdl.nn.keras.Sequential" ->
      "com.intel.analytics.bigdl.keras.nn.SequentialSerialTest",
    "com.intel.analytics.bigdl.nn.keras.Activation" ->
      "com.intel.analytics.bigdl.keras.nn.ActivationSerialTest",
    "com.intel.analytics.bigdl.nn.keras.SoftMax" ->
      "com.intel.analytics.bigdl.keras.nn.SoftMaxSerialTest",
    "com.intel.analytics.bigdl.nn.keras.AtrousConvolution1D" ->
      "com.intel.analytics.bigdl.keras.nn.AtrousConvolution1DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.AtrousConvolution2D" ->
      "com.intel.analytics.bigdl.keras.nn.AtrousConvolution2DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.AveragePooling1D" ->
      "com.intel.analytics.bigdl.keras.nn.AveragePooling1DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.AveragePooling2D" ->
      "com.intel.analytics.bigdl.keras.nn.AveragePooling2DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.AveragePooling3D" ->
      "com.intel.analytics.bigdl.keras.nn.AveragePooling3DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.BatchNormalization" ->
      "com.intel.analytics.bigdl.keras.nn.BatchNormalizationSerialTest",
    "com.intel.analytics.bigdl.nn.keras.Bidirectional" ->
      "com.intel.analytics.bigdl.keras.nn.BidirectionalSerialTest",
    "com.intel.analytics.bigdl.nn.keras.ConvLSTM2D" ->
      "com.intel.analytics.bigdl.keras.nn.ConvLSTM2DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.Convolution1D" ->
      "com.intel.analytics.bigdl.keras.nn.Convolution1DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.Convolution2D" ->
      "com.intel.analytics.bigdl.keras.nn.Convolution2DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.Convolution3D" ->
      "com.intel.analytics.bigdl.keras.nn.Convolution3DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.Cropping1D" ->
      "com.intel.analytics.bigdl.keras.nn.Cropping1DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.Cropping2D" ->
      "com.intel.analytics.bigdl.keras.nn.Cropping2DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.Deconvolution2D" ->
      "com.intel.analytics.bigdl.keras.nn.Deconvolution2DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.ELU" ->
      "com.intel.analytics.bigdl.keras.nn.ELUSerialTest",
    "com.intel.analytics.bigdl.nn.keras.Embedding" ->
      "com.intel.analytics.bigdl.keras.nn.EmbeddingSerialTest",
    "com.intel.analytics.bigdl.nn.keras.GaussianDropout" ->
      "com.intel.analytics.bigdl.keras.nn.GaussianDropoutSerialTest",
    "com.intel.analytics.bigdl.nn.keras.GaussianNoise" ->
      "com.intel.analytics.bigdl.keras.nn.GaussianNoiseSerialTest",
    "com.intel.analytics.bigdl.nn.keras.GlobalAveragePooling2D" ->
      "com.intel.analytics.bigdl.keras.nn.GlobalAveragePooling2DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.GlobalMaxPooling2D" ->
      "com.intel.analytics.bigdl.keras.nn.GlobalMaxPooling2DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.GlobalMaxPooling3D" ->
      "com.intel.analytics.bigdl.keras.nn.GlobalMaxPooling3DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.GRU" ->
      "com.intel.analytics.bigdl.keras.nn.GRUSerialTest",
    "com.intel.analytics.bigdl.nn.keras.Highway" ->
      "com.intel.analytics.bigdl.keras.nn.HighwaySerialTest",
    "com.intel.analytics.bigdl.nn.keras.LeakyReLU" ->
      "com.intel.analytics.bigdl.keras.nn.LeakyReLUSerialTest",
    "com.intel.analytics.bigdl.nn.keras.LocallyConnected1D" ->
      "com.intel.analytics.bigdl.keras.nn.LocallyConnected1DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.LocallyConnected2D" ->
      "com.intel.analytics.bigdl.keras.nn.LocallyConnected2DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.LSTM" ->
      "com.intel.analytics.bigdl.keras.nn.LSTMSerialTest",
    "com.intel.analytics.bigdl.nn.keras.Masking" ->
      "com.intel.analytics.bigdl.keras.nn.MaskingSerialTest",
    "com.intel.analytics.bigdl.nn.keras.MaxoutDense" ->
      "com.intel.analytics.bigdl.keras.nn.MaxoutDenseSerialTest",
    "com.intel.analytics.bigdl.nn.keras.MaxPooling1D" ->
      "com.intel.analytics.bigdl.keras.nn.MaxPooling1DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.MaxPooling2D" ->
      "com.intel.analytics.bigdl.keras.nn.MaxPooling2DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.MaxPooling3D" ->
      "com.intel.analytics.bigdl.keras.nn.MaxPooling3DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.Merge" ->
      "com.intel.analytics.bigdl.keras.nn.MergeSerialTest",
    "com.intel.analytics.bigdl.nn.keras.RepeatVector" ->
      "com.intel.analytics.bigdl.keras.nn.RepeatVectorSerialTest",
    "com.intel.analytics.bigdl.nn.keras.SeparableConvolution2D" ->
      "com.intel.analytics.bigdl.keras.nn.SeparableConvolution2DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.SimpleRNN" ->
      "com.intel.analytics.bigdl.keras.nn.SimpleRNNSerialTest",
    "com.intel.analytics.bigdl.nn.keras.SpatialDropout1D" ->
      "com.intel.analytics.bigdl.keras.nn.SpatialDropout1DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.SpatialDropout2D" ->
      "com.intel.analytics.bigdl.keras.nn.SpatialDropout2DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.SpatialDropout3D" ->
      "com.intel.analytics.bigdl.keras.nn.SpatialDropout3DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.SReLU" ->
      "com.intel.analytics.bigdl.keras.nn.SReLUSerialTest",
    "com.intel.analytics.bigdl.nn.keras.ThresholdedReLU" ->
      "com.intel.analytics.bigdl.keras.nn.ThresholdedReLUSerialTest",
    "com.intel.analytics.bigdl.nn.keras.TimeDistributed" ->
      "com.intel.analytics.bigdl.keras.nn.TimeDistributedSerialTest",
    "com.intel.analytics.bigdl.nn.keras.UpSampling1D" ->
      "com.intel.analytics.bigdl.keras.nn.UpSampling1DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.UpSampling2D" ->
      "com.intel.analytics.bigdl.keras.nn.UpSampling2DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.UpSampling3D" ->
      "com.intel.analytics.bigdl.keras.nn.UpSampling3DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.ZeroPadding1D" ->
      "com.intel.analytics.bigdl.keras.nn.ZeroPadding1DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.ZeroPadding2D" ->
      "com.intel.analytics.bigdl.keras.nn.ZeroPadding2DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.Dense" ->
      "com.intel.analytics.bigdl.keras.nn.DenseSerialTest",
    "com.intel.analytics.bigdl.nn.keras.Cropping3D" ->
      "com.intel.analytics.bigdl.keras.nn.Cropping3DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.Reshape" ->
      "com.intel.analytics.bigdl.keras.nn.ReshapeSerialTest",
    "com.intel.analytics.bigdl.nn.keras.Permute" ->
      "com.intel.analytics.bigdl.keras.nn.PermuteSerialTest",
    "com.intel.analytics.bigdl.nn.keras.Model" ->
      "com.intel.analytics.bigdl.keras.nn.ModelSerialTest",
    "com.intel.analytics.bigdl.nn.keras.GlobalAveragePooling3D" ->
      "com.intel.analytics.bigdl.keras.nn.GlobalAveragePooling3DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.GlobalAveragePooling1D" ->
      "com.intel.analytics.bigdl.keras.nn.GlobalAveragePooling1DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.ZeroPadding3D" ->
      "com.intel.analytics.bigdl.keras.nn.ZeroPadding3DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.Dropout" ->
      "com.intel.analytics.bigdl.keras.nn.DropoutSerialTest",
    "com.intel.analytics.bigdl.nn.keras.GlobalMaxPooling1D" ->
      "com.intel.analytics.bigdl.keras.nn.GlobalMaxPooling1DSerialTest",
    "com.intel.analytics.bigdl.nn.keras.Flatten" ->
      "com.intel.analytics.bigdl.keras.nn.FlattenSerialTest",
  "com.intel.analytics.bigdl.nn.keras.KerasIdentityWrapper" ->
    "com.intel.analytics.bigdl.keras.nn.KerasIdentityWrapperSerialTest",
    "com.intel.analytics.bigdl.nn.keras.KerasLayerWrapper" ->
      "com.intel.analytics.bigdl.keras.nn.KerasLayerWrapperSerialTest"
  )

  private val suffix = "SerialTest"

  private val testClasses = new mutable.HashSet[String]()

  {
    val filterBuilder = new FilterBuilder()
    val reflections = new Reflections(new ConfigurationBuilder()
      .filterInputsBy(filterBuilder)
      .setUrls(ClasspathHelper.forPackage("com.intel.analytics.bigdl.nn"))
      .setScanners(new SubTypesScanner()))


    val subTypes = reflections.getSubTypesOf(classOf[AbstractModule[_, _, _]])
      .asScala.filter(sub => !Modifier.isAbstract(sub.getModifiers))
      .filter(sub => !excluded.contains(sub.getName))
    subTypes.foreach(sub => testClasses.add(sub.getName))
  }

  private def getTestClassName(clsName: String): String = {
    if (unRegularNameMapping.contains(clsName)) {
      unRegularNameMapping(clsName)
    } else {
      clsName + suffix
    }
  }

  testClasses.foreach(cls => {
    "Serialization test of module " + cls should "be correct" in {
      val clsWholeName = getTestClassName(cls)
      try {
        val ins = Class.forName(clsWholeName)
        val testClass = ins.getConstructors()(0).newInstance()
        require(testClass.isInstanceOf[ModuleSerializationTest], s"$clsWholeName should be a " +
          s"subclass of com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest")
        testClass.asInstanceOf[ModuleSerializationTest].test()
      } catch {
        case t: Throwable => throw t
      }
    }
  })

  "Group serializer" should "work properly" in {
    ModuleSerializer.
      registerGroupModules("com.intel.analytics.bigdl.utils.serializer.ParentModule",
      ParentModuleSerializer)
    val subOne = new SubModuleOne[Float]()
    val subTwo = new SubModuleTwo[Float]()
    val serFileOne = File.createTempFile("SubOne", "bigdl")
    val serFileTwo = File.createTempFile("SubTwo", "bigdl")
    subOne.saveModule(serFileOne.getAbsolutePath, overWrite = true)
    subTwo.saveModule(serFileTwo.getAbsolutePath, overWrite = true)

    val loadedOne = Module.loadModule[Float](serFileOne.getAbsolutePath).
      asInstanceOf[SubModuleOne[Float]]

    val loadedTwo = Module.loadModule[Float](serFileTwo.getAbsolutePath).
      asInstanceOf[SubModuleTwo[Float]]

    loadedOne.value should be ("test_value")

    loadedTwo.value should be ("test_value")
  }
}

abstract class ParentModule[T: ClassTag](implicit ev: TensorNumeric[T]) extends
  AbstractModule[Tensor[T], Tensor[T], T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    null
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    null
  }

  var value : String = null
}

class SubModuleOne[T: ClassTag](implicit ev: TensorNumeric[T]) extends ParentModule[T] {

}

class SubModuleTwo[T: ClassTag](implicit ev: TensorNumeric[T]) extends ParentModule[T] {

}

object ParentModuleSerializer extends ModuleSerializable {
  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
  bigDLModelBuilder: BigDLModule.Builder)(implicit ev: TensorNumericMath.TensorNumeric[T]): Unit = {
    val groupTypeAttrValue = AttrValue.newBuilder
    DataConverter.setAttributeValue[T](context, groupTypeAttrValue,
      "test_value", universe.typeOf[String])
    bigDLModelBuilder.putAttr("groupValue", groupTypeAttrValue.build)
  }

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumericMath.TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {
    val module = super.doLoadModule(context).asInstanceOf[ParentModule[T]]
    val attrMap = context.bigdlModule.getAttrMap
    val valueAttr = attrMap.get("groupValue")
    val value = DataConverter.getAttributeValue(context, valueAttr).
      asInstanceOf[String]
    module.value = value
    module
  }
}

private[bigdl] abstract class ModuleSerializationTest extends SerializerSpecHelper {
  def test(): Unit
}
