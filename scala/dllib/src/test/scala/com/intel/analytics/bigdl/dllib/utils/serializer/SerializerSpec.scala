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
package com.intel.analytics.bigdl.dllib.utils.serializer

import java.io.File
import java.lang.reflect.Modifier

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dllib.nn.Module
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.tensor.{Tensor, TensorNumericMath}
import com.intel.analytics.bigdl.dllib.utils.BigDLSpecHelper
import com.intel.analytics.bigdl.dllib.utils.serializer.converters.DataConverter
import org.reflections.Reflections
import org.reflections.scanners.SubTypesScanner
import org.reflections.util.{ClasspathHelper, ConfigurationBuilder, FilterBuilder}

import collection.JavaConverters._
import scala.collection.mutable
import scala.reflect.ClassTag
import scala.reflect.runtime.universe

class SerializerSpec extends BigDLSpecHelper {

  private val excluded = Set[String](
    "com.intel.analytics.bigdl.dllib.nn.CellUnit",
    "com.intel.analytics.bigdl.dllib.nn.tf.ControlDependency",
    "com.intel.analytics.bigdl.dllib.utils.tf.AdapterForTest",
    "com.intel.analytics.bigdl.dllib.utils.serializer.TestModule",
    "com.intel.analytics.bigdl.dllib.utils.TestModule",
    "com.intel.analytics.bigdl.dllib.utils.ExceptionTest",
    "com.intel.analytics.bigdl.dllib.utils.serializer.SubModuleOne",
    "com.intel.analytics.bigdl.dllib.utils.serializer.SubModuleTwo",
    "com.intel.analytics.bigdl.dllib.nn.mkldnn.AvgPooling",
    "com.intel.analytics.bigdl.dllib.nn.mkldnn.CAddTable",
    "com.intel.analytics.bigdl.dllib.nn.mkldnn.ConcatTable",
    "com.intel.analytics.bigdl.dllib.nn.mkldnn.DnnBase",
    "com.intel.analytics.bigdl.dllib.nn.mkldnn.Identity",
    "com.intel.analytics.bigdl.dllib.nn.mkldnn.Input",
    "com.intel.analytics.bigdl.dllib.nn.mkldnn.JoinTable",
    "com.intel.analytics.bigdl.dllib.nn.mkldnn.Linear",
    "com.intel.analytics.bigdl.dllib.nn.mkldnn.LRN",
    "com.intel.analytics.bigdl.dllib.nn.mkldnn.MaxPooling",
    "com.intel.analytics.bigdl.dllib.nn.mkldnn.ReLU",
    "com.intel.analytics.bigdl.dllib.nn.mkldnn.ReorderMemory",
    "com.intel.analytics.bigdl.dllib.nn.mkldnn.SelectTable",
    "com.intel.analytics.bigdl.dllib.nn.mkldnn.Sequential",
    "com.intel.analytics.bigdl.dllib.nn.mkldnn.SoftMax",
    "com.intel.analytics.bigdl.dllib.nn.mkldnn.SpatialBatchNormalization",
    "com.intel.analytics.bigdl.dllib.nn.mkldnn.SpatialConvolution",
    "com.intel.analytics.bigdl.dllib.nn.mkldnn.Dropout",
    "com.intel.analytics.bigdl.dllib.nn.mkldnn.DnnGraph",
    "com.intel.analytics.bigdl.dllib.nn.mkldnn.BlasWrapper",
    "com.intel.analytics.bigdl.dllib.nn.mkldnn.Output",
    "com.intel.analytics.bigdl.dllib.nn.mkldnn.InputWrapper",
    "com.intel.analytics.bigdl.dllib.utils.intermediate.IRGraph",
    "com.intel.analytics.bigdl.dllib.nn.mkldnn.RNN",
    "com.intel.analytics.bigdl.dllib.nn.ops.TruncatedNormal",
    "com.intel.analytics.bigdl.dllib.keras.layers.Input",
    "com.intel.analytics.bigdl.dllib.keras.layers.Sequential",
    "com.intel.analytics.bigdl.dllib.keras.layers.Dense",
    "com.intel.analytics.bigdl.dllib.keras.layers.ZeroPadding1D",
    "com.intel.analytics.bigdl.dllib.keras.layers.Scale",
    "com.intel.analytics.bigdl.dllib.keras.layers.ZeroPadding1D",
    "com.intel.analytics.bigdl.dllib.keras.layers.TransformerLayer",
    "com.intel.analytics.bigdl.dllib.keras.layers.WithinChannelLRN2D",
    "com.intel.analytics.bigdl.dllib.keras.layers.Cropping2D",
    "com.intel.analytics.bigdl.dllib.keras.layers.HardShrink",
    "com.intel.analytics.bigdl.dllib.keras.layers.UpSampling1D",
    "com.intel.analytics.bigdl.dllib.keras.layers.Sqrt",
    "com.intel.analytics.bigdl.dllib.keras.layers.ThresholdedReLU",
    "com.intel.analytics.bigdl.dllib.keras.layers.Convolution2D",
    "com.intel.analytics.bigdl.dllib.keras.layers.SimpleRNN",
    "com.intel.analytics.bigdl.dllib.keras.layers.SoftShrink",
    "com.intel.analytics.bigdl.dllib.keras.layers.SoftMax",
    "com.intel.analytics.bigdl.dllib.keras.layers.RReLU",
    "com.intel.analytics.bigdl.dllib.keras.layers.Convolution3D",
    "com.intel.analytics.bigdl.dllib.keras.layers.Exp",
    "com.intel.analytics.bigdl.dllib.keras.layers.GetShape",
    "com.intel.analytics.bigdl.dllib.keras.layers.SelectTable",
    "com.intel.analytics.bigdl.dllib.keras.layers.Highway",
    "com.intel.analytics.bigdl.dllib.keras.layers.TimeDistributed",
    "com.intel.analytics.bigdl.dllib.keras.layers.KerasRunner",
    "com.intel.analytics.bigdl.dllib.keras.layers.GaussianDropout",
    "com.intel.analytics.bigdl.dllib.keras.layers.Identity",
    "com.intel.analytics.bigdl.dllib.keras.layers.LSTM",
    "com.intel.analytics.bigdl.dllib.keras.layers.Masking",
    "com.intel.analytics.bigdl.dllib.keras.layers.Narrow",
    "com.intel.analytics.bigdl.dllib.keras.layers.ResizeBilinear",
    "com.intel.analytics.bigdl.dllib.keras.layers.CAdd",
    "com.intel.analytics.bigdl.dllib.keras.layers.MulConstant",
    "com.intel.analytics.bigdl.dllib.keras.layers.BinaryThreshold",
    "com.intel.analytics.bigdl.dllib.keras.layers.HardTanh",
    "com.intel.analytics.bigdl.dllib.keras.layers.ConvLSTM2D",
    "com.intel.analytics.bigdl.dllib.keras.layers.InternalConvLSTM3D",
    "com.intel.analytics.bigdl.dllib.keras.layers.GlobalAveragePooling1D",
    "com.intel.analytics.bigdl.dllib.keras.layers.Max",
    "com.intel.analytics.bigdl.dllib.keras.layers.Squeeze",
    "com.intel.analytics.bigdl.dllib.keras.layers.LRN2D",
    "com.intel.analytics.bigdl.dllib.keras.layers.InternalGetShape",
    "com.intel.analytics.bigdl.dllib.keras.layers.Reshape",
    "com.intel.analytics.bigdl.dllib.keras.layers.LayerNorm",
    "com.intel.analytics.bigdl.dllib.keras.layers.AtrousConvolution1D",
    "com.intel.analytics.bigdl.dllib.keras.layers.Bert",
    "com.intel.analytics.bigdl.dllib.keras.layers.Power",
    "com.intel.analytics.bigdl.dllib.keras.layers.SpatialDropout3D",
    "com.intel.analytics.bigdl.dllib.keras.layers.AtrousConvolution2D",
    "com.intel.analytics.bigdl.dllib.keras.layers.LeakyReLU",
    "com.intel.analytics.bigdl.dllib.keras.layers.Square",
    "com.intel.analytics.bigdl.dllib.keras.layers.KerasLayerWrapper",
    "com.intel.analytics.bigdl.dllib.keras.layers.ELU",
    "com.intel.analytics.bigdl.dllib.keras.layers.GlobalMaxPooling1D",
    "com.intel.analytics.bigdl.dllib.keras.layers.AveragePooling1D",
    "com.intel.analytics.bigdl.dllib.keras.layers.Bidirectional",
    "com.intel.analytics.bigdl.dllib.keras.layers.Negative",
    "com.intel.analytics.bigdl.dllib.keras.layers.PReLU",
    "com.intel.analytics.bigdl.dllib.keras.layers.GaussianSampler",
    "com.intel.analytics.bigdl.dllib.keras.layers.MaxPooling3D",
    "com.intel.analytics.bigdl.dllib.keras.layers.SReLU",
    "com.intel.analytics.bigdl.dllib.keras.layers.UpSampling2D",
    "com.intel.analytics.bigdl.dllib.keras.layers.Activation",
    "com.intel.analytics.bigdl.dllib.keras.layers.Dropout",
    "com.intel.analytics.bigdl.dllib.keras.layers.Log",
    "com.intel.analytics.bigdl.dllib.keras.layers.LocallyConnected1D",
    "com.intel.analytics.bigdl.dllib.keras.layers.ExpandDim",
    "com.intel.analytics.bigdl.dllib.keras.layers.GRU",
    "com.intel.analytics.bigdl.dllib.keras.layers.BatchNormalization",
    "com.intel.analytics.bigdl.dllib.keras.layers.KerasBase",
    "com.intel.analytics.bigdl.dllib.keras.layers.SpatialDropout1D",
    "com.intel.analytics.bigdl.dllib.keras.layers.ShareConvolution2D",
    "com.intel.analytics.bigdl.dllib.keras.layers.MaxPooling2D",
    "com.intel.analytics.bigdl.dllib.keras.layers.SparseEmbedding",
    "com.intel.analytics.bigdl.dllib.keras.layers.SeparableConvolution2D",
    "com.intel.analytics.bigdl.dllib.keras.layers.Convolution1D",
    "com.intel.analytics.bigdl.dllib.keras.layers.LocallyConnected2D",
    "com.intel.analytics.bigdl.dllib.keras.layers.Select",
    "com.intel.analytics.bigdl.dllib.keras.layers.InternalExpand",
    "com.intel.analytics.bigdl.dllib.keras.layers.RepeatVector",
    "com.intel.analytics.bigdl.dllib.keras.layers.GlobalAveragePooling2D",
    "com.intel.analytics.bigdl.dllib.keras.layers.ZeroPadding3D",
    "com.intel.analytics.bigdl.dllib.keras.layers.Mul",
    "com.intel.analytics.bigdl.dllib.keras.layers.UpSampling3D",
    "com.intel.analytics.bigdl.dllib.keras.layers.WordEmbedding",
    "com.intel.analytics.bigdl.dllib.keras.layers.Deconvolution2D",
    "com.intel.analytics.bigdl.dllib.keras.layers.MaxPooling1D",
    "com.intel.analytics.bigdl.dllib.keras.layers.Cropping3D",
    "com.intel.analytics.bigdl.dllib.keras.layers.Cropping1D",
    "com.intel.analytics.bigdl.dllib.keras.layers.MaxoutDense",
    "com.intel.analytics.bigdl.dllib.keras.layers.Merge",
    "com.intel.analytics.bigdl.dllib.keras.layers.Parameter",
    "com.intel.analytics.bigdl.dllib.keras.layers.GlobalMaxPooling2D",
    "com.intel.analytics.bigdl.dllib.keras.layers.AveragePooling3D",
    "com.intel.analytics.bigdl.dllib.keras.layers.Flatten",
    "com.intel.analytics.bigdl.dllib.keras.layers.InternalConvLSTM2D",
    "com.intel.analytics.bigdl.dllib.keras.layers.SpatialDropout2D",
    "com.intel.analytics.bigdl.dllib.keras.layers.Expand",
    "com.intel.analytics.bigdl.dllib.keras.layers.Embedding",
    "com.intel.analytics.bigdl.dllib.keras.layers.AveragePooling2D",
    "com.intel.analytics.bigdl.dllib.keras.layers.ZeroPadding2D",
    "com.intel.analytics.bigdl.dllib.keras.layers.GlobalMaxPooling3D",
    "com.intel.analytics.bigdl.dllib.keras.layers.AddConstant",
    "com.intel.analytics.bigdl.dllib.keras.layers.GaussianNoise",
    "com.intel.analytics.bigdl.dllib.keras.layers.SparseDense",
    "com.intel.analytics.bigdl.dllib.keras.layers.ConvLSTM3D",
    "com.intel.analytics.bigdl.dllib.keras.layers.CMul",
    "com.intel.analytics.bigdl.dllib.keras.layers.Softmax",
    "com.intel.analytics.bigdl.dllib.keras.layers.GlobalAveragePooling3D",
    "com.intel.analytics.bigdl.dllib.keras.layers.Threshold",
    "com.intel.analytics.bigdl.dllib.keras.layers.Permute",
    "com.intel.analytics.bigdl.dllib.keras.layers.BERT",
    "com.intel.analytics.bigdl.dllib.keras.layers.SplitTensor",
    "com.intel.analytics.bigdl.dllib.keras.layers.internal.InternalExpand",
    "com.intel.analytics.bigdl.dllib.keras.layers.internal.InternalTimeDistributed",
    "com.intel.analytics.bigdl.dllib.keras.layers.internal.InternalRecurrent",
    "com.intel.analytics.bigdl.dllib.keras.layers.internal.InternalSoftmax",
    "com.intel.analytics.bigdl.dllib.keras.layers.internal.InternalCMulTable",
    "com.intel.analytics.bigdl.dllib.keras.layers.internal.InternalLayerNorm",
    "com.intel.analytics.bigdl.dllib.keras.layers.internal.InternalMax",
    "com.intel.analytics.bigdl.dllib.keras.layers.internal.InternalSplitTensor",
    "com.intel.analytics.bigdl.dllib.keras.layers.internal.InternalMM",
    "com.intel.analytics.bigdl.dllib.keras.layers.internal.InternalERF",
    "com.intel.analytics.bigdl.dllib.keras.layers.internal.InternalCAddTable",
    "com.intel.analytics.bigdl.dllib.keras.autograd.KerasConstant",
    "com.intel.analytics.bigdl.dllib.keras.autograd.KerasParameter",
    "com.intel.analytics.bigdl.dllib.keras.autograd.LambdaTorch",
    "com.intel.analytics.bigdl.dllib.keras.autograd.InternalParameter",
    "com.intel.analytics.bigdl.dllib.keras.autograd.LambdaLayer",
    "com.intel.analytics.bigdl.dllib.keras.autograd.InternalConstant",
    "com.intel.analytics.bigdl.dllib.keras.Model",
    "com.intel.analytics.bigdl.dllib.keras.Sequential",
    "com.intel.analytics.bigdl.dllib.net.GraphNet",
    "com.intel.analytics.bigdl.dllib.nn.keras.Sequential",
    "com.intel.analytics.bigdl.dllib.nn.keras.LocallyConnected2D",
    "com.intel.analytics.bigdl.dllib.nn.keras.Input",
    "com.intel.analytics.bigdl.dllib.nn.keras.Model"
  )

  // Maybe one serial test class contains multiple module test
  // Also keras layer main/test class mapping are weired
  private val unRegularNameMapping = Map[String, String](
    // Many to one mapping
    "com.intel.analytics.bigdl.dllib.nn.ops.Enter" ->
      "com.intel.analytics.bigdl.dllib.nn.ops.ControlOpsSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.tf.Enter" ->
      "com.intel.analytics.bigdl.dllib.nn.tf.ControlOpsSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.ops.NextIteration" ->
      "com.intel.analytics.bigdl.dllib.nn.ops.ControlOpsSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.tf.NextIteration" ->
      "com.intel.analytics.bigdl.dllib.nn.tf.ControlOpsSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.ops.Exit" ->
      "com.intel.analytics.bigdl.dllib.nn.ops.ControlOpsSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.tf.Exit" ->
      "com.intel.analytics.bigdl.dllib.nn.tf.ControlOpsSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.ops.LoopCondition" ->
      "com.intel.analytics.bigdl.dllib.nn.ops.ControlOpsSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.tf.LoopCondition" ->
      "com.intel.analytics.bigdl.dllib.nn.tf.ControlOpsSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.ops.StackCreator" ->
      "com.intel.analytics.bigdl.dllib.nn.ops.StackOpsSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.tf.StackCreator" ->
      "com.intel.analytics.bigdl.dllib.nn.tf.StackOpsSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.ops.StackPush" ->
      "com.intel.analytics.bigdl.dllib.nn.ops.StackOpsSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.tf.StackPush" ->
      "com.intel.analytics.bigdl.dllib.nn.tf.StackOpsSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.ops.StackPop" ->
      "com.intel.analytics.bigdl.dllib.nn.ops.StackOpsSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.tf.StackPop" ->
      "com.intel.analytics.bigdl.dllib.nn.tf.StackOpsSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.ops.TensorArrayWrite" ->
      "com.intel.analytics.bigdl.dllib.nn.ops.TensorArraySerialTest",
    "com.intel.analytics.bigdl.dllib.nn.tf.TensorArrayWrite" ->
      "com.intel.analytics.bigdl.dllib.nn.ops.TensorArraySerialTest",
    "com.intel.analytics.bigdl.dllib.nn.ops.TensorArrayRead" ->
      "com.intel.analytics.bigdl.dllib.nn.ops.TensorArraySerialTest",
    "com.intel.analytics.bigdl.dllib.nn.tf.TensorArrayRead" ->
      "com.intel.analytics.bigdl.dllib.nn.ops.TensorArraySerialTest",
    "com.intel.analytics.bigdl.dllib.nn.ops.TensorArrayGrad" ->
      "com.intel.analytics.bigdl.dllib.nn.ops.TensorArraySerialTest",
    "com.intel.analytics.bigdl.dllib.nn.tf.TensorArrayGrad" ->
      "com.intel.analytics.bigdl.dllib.nn.ops.TensorArraySerialTest",
    "com.intel.analytics.bigdl.dllib.nn.tf.TensorArrayCreator" ->
      "com.intel.analytics.bigdl.dllib.nn.ops.TensorArrayScatterSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.tf.TensorArrayScatter" ->
      "com.intel.analytics.bigdl.dllib.nn.ops.TensorArrayScatterSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.tf.TensorArrayGather" ->
      "com.intel.analytics.bigdl.dllib.nn.ops.TensorArrayScatterSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.tf.TensorArrayClose" ->
      "com.intel.analytics.bigdl.dllib.nn.ops.TensorArrayScatterSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.tf.TensorArrayConcat" ->
      "com.intel.analytics.bigdl.dllib.nn.ops.TensorArraySplitSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.tf.TensorArraySplit" ->
      "com.intel.analytics.bigdl.dllib.nn.ops.TensorArraySplitSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.tf.TensorArraySize" ->
      "com.intel.analytics.bigdl.dllib.nn.ops.TensorArraySplitSerialTest",

    // Keras layers
    "com.intel.analytics.bigdl.dllib.nn.keras.Input" ->
      "com.intel.analytics.bigdl.keras.nn.InputSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.Sequential" ->
      "com.intel.analytics.bigdl.keras.nn.SequentialSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.Activation" ->
      "com.intel.analytics.bigdl.keras.nn.ActivationSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.SoftMax" ->
      "com.intel.analytics.bigdl.keras.nn.SoftMaxSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.AtrousConvolution1D" ->
      "com.intel.analytics.bigdl.keras.nn.AtrousConvolution1DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.AtrousConvolution2D" ->
      "com.intel.analytics.bigdl.keras.nn.AtrousConvolution2DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.AveragePooling1D" ->
      "com.intel.analytics.bigdl.keras.nn.AveragePooling1DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.AveragePooling2D" ->
      "com.intel.analytics.bigdl.keras.nn.AveragePooling2DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.AveragePooling3D" ->
      "com.intel.analytics.bigdl.keras.nn.AveragePooling3DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.BatchNormalization" ->
      "com.intel.analytics.bigdl.keras.nn.BatchNormalizationSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.Bidirectional" ->
      "com.intel.analytics.bigdl.keras.nn.BidirectionalSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.ConvLSTM2D" ->
      "com.intel.analytics.bigdl.keras.nn.ConvLSTM2DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.Convolution1D" ->
      "com.intel.analytics.bigdl.keras.nn.Convolution1DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.Convolution2D" ->
      "com.intel.analytics.bigdl.keras.nn.Convolution2DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.Convolution3D" ->
      "com.intel.analytics.bigdl.keras.nn.Convolution3DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.Cropping1D" ->
      "com.intel.analytics.bigdl.keras.nn.Cropping1DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.Cropping2D" ->
      "com.intel.analytics.bigdl.keras.nn.Cropping2DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.Deconvolution2D" ->
      "com.intel.analytics.bigdl.keras.nn.Deconvolution2DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.ELU" ->
      "com.intel.analytics.bigdl.keras.nn.ELUSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.Embedding" ->
      "com.intel.analytics.bigdl.keras.nn.EmbeddingSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.GaussianDropout" ->
      "com.intel.analytics.bigdl.keras.nn.GaussianDropoutSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.GaussianNoise" ->
      "com.intel.analytics.bigdl.keras.nn.GaussianNoiseSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.GlobalAveragePooling2D" ->
      "com.intel.analytics.bigdl.keras.nn.GlobalAveragePooling2DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.GlobalMaxPooling2D" ->
      "com.intel.analytics.bigdl.keras.nn.GlobalMaxPooling2DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.GlobalMaxPooling3D" ->
      "com.intel.analytics.bigdl.keras.nn.GlobalMaxPooling3DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.GRU" ->
      "com.intel.analytics.bigdl.keras.nn.GRUSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.Highway" ->
      "com.intel.analytics.bigdl.keras.nn.HighwaySerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.LeakyReLU" ->
      "com.intel.analytics.bigdl.keras.nn.LeakyReLUSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.LocallyConnected1D" ->
      "com.intel.analytics.bigdl.keras.nn.LocallyConnected1DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.LocallyConnected2D" ->
      "com.intel.analytics.bigdl.keras.nn.LocallyConnected2DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.LSTM" ->
      "com.intel.analytics.bigdl.keras.nn.LSTMSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.Masking" ->
      "com.intel.analytics.bigdl.keras.nn.MaskingSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.MaxoutDense" ->
      "com.intel.analytics.bigdl.keras.nn.MaxoutDenseSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.MaxPooling1D" ->
      "com.intel.analytics.bigdl.keras.nn.MaxPooling1DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.MaxPooling2D" ->
      "com.intel.analytics.bigdl.keras.nn.MaxPooling2DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.MaxPooling3D" ->
      "com.intel.analytics.bigdl.keras.nn.MaxPooling3DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.Merge" ->
      "com.intel.analytics.bigdl.keras.nn.MergeSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.RepeatVector" ->
      "com.intel.analytics.bigdl.keras.nn.RepeatVectorSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.SeparableConvolution2D" ->
      "com.intel.analytics.bigdl.keras.nn.SeparableConvolution2DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.SimpleRNN" ->
      "com.intel.analytics.bigdl.keras.nn.SimpleRNNSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.SpatialDropout1D" ->
      "com.intel.analytics.bigdl.keras.nn.SpatialDropout1DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.SpatialDropout2D" ->
      "com.intel.analytics.bigdl.keras.nn.SpatialDropout2DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.SpatialDropout3D" ->
      "com.intel.analytics.bigdl.keras.nn.SpatialDropout3DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.SReLU" ->
      "com.intel.analytics.bigdl.keras.nn.SReLUSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.ThresholdedReLU" ->
      "com.intel.analytics.bigdl.keras.nn.ThresholdedReLUSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.TimeDistributed" ->
      "com.intel.analytics.bigdl.keras.nn.TimeDistributedSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.UpSampling1D" ->
      "com.intel.analytics.bigdl.keras.nn.UpSampling1DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.UpSampling2D" ->
      "com.intel.analytics.bigdl.keras.nn.UpSampling2DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.UpSampling3D" ->
      "com.intel.analytics.bigdl.keras.nn.UpSampling3DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.ZeroPadding1D" ->
      "com.intel.analytics.bigdl.keras.nn.ZeroPadding1DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.ZeroPadding2D" ->
      "com.intel.analytics.bigdl.keras.nn.ZeroPadding2DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.Dense" ->
      "com.intel.analytics.bigdl.keras.nn.DenseSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.Cropping3D" ->
      "com.intel.analytics.bigdl.keras.nn.Cropping3DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.Reshape" ->
      "com.intel.analytics.bigdl.keras.nn.ReshapeSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.Permute" ->
      "com.intel.analytics.bigdl.keras.nn.PermuteSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.Model" ->
      "com.intel.analytics.bigdl.keras.nn.ModelSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.GlobalAveragePooling3D" ->
      "com.intel.analytics.bigdl.keras.nn.GlobalAveragePooling3DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.GlobalAveragePooling1D" ->
      "com.intel.analytics.bigdl.keras.nn.GlobalAveragePooling1DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.ZeroPadding3D" ->
      "com.intel.analytics.bigdl.keras.nn.ZeroPadding3DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.Dropout" ->
      "com.intel.analytics.bigdl.keras.nn.DropoutSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.GlobalMaxPooling1D" ->
      "com.intel.analytics.bigdl.keras.nn.GlobalMaxPooling1DSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.Flatten" ->
      "com.intel.analytics.bigdl.keras.nn.FlattenSerialTest",
  "com.intel.analytics.bigdl.dllib.nn.keras.KerasIdentityWrapper" ->
    "com.intel.analytics.bigdl.keras.nn.KerasIdentityWrapperSerialTest",
    "com.intel.analytics.bigdl.dllib.nn.keras.KerasLayerWrapper" ->
      "com.intel.analytics.bigdl.keras.nn.KerasLayerWrapperSerialTest"
  )

  private val suffix = "SerialTest"

  private val testClasses = new mutable.HashSet[String]()

  {
    val filterBuilder = new FilterBuilder()
    val reflections = new Reflections(new ConfigurationBuilder()
      .filterInputsBy(filterBuilder)
      .setUrls(ClasspathHelper.forPackage("com.intel.analytics.bigdl.dllib.nn"))
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
          s"subclass of com.intel.analytics.bigdl.dllib.utils.serializer.ModuleSerializationTest")
        testClass.asInstanceOf[ModuleSerializationTest].test()
      } catch {
        case t: Throwable => throw t
      }
    }
  })

  "Group serializer" should "work properly" in {
    ModuleSerializer.
      registerGroupModules("com.intel.analytics.bigdl.dllib.utils.serializer.ParentModule",
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
