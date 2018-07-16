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

package com.intel.analytics.bigdl.example.recommendation

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import org.apache.spark.sql.catalyst.plans.physical.IdentityBroadcastMode

import scala.reflect.ClassTag

/**
 * The model is for neural collaborative filtering.
 *
 * @param numClasses   The number of classes. Positive integer.
 * @param userCount    The number of users. Positive integer.
 * @param itemCount    The number of items. Positive integer.
 * @param userEmbed    Units of user embedding. Positive integer.
 * @param itemEmbed    Units of item embedding. Positive integer.
 * @param hiddenLayers Units hidenLayers of MLP part. Array of positive integer.
 * @param includeMF    Include Matrix Factorization or not. Boolean.
 * @param mfEmbed      Units of matrix factorization embedding. Positive integer.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */

class NeuralCFV2[T: ClassTag] private(val userCount: Int,
       val itemCount: Int,
       val numClasses: Int,
       val userEmbed: Int = 20,
       val itemEmbed: Int = 20,
       val hiddenLayers: Array[Int] = Array(40, 20, 10),
       val includeMF: Boolean = true,
       val mfEmbed: Int = 20
      )(implicit ev: TensorNumeric[T]) extends Container[Tensor[T], Tensor[T], T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output = ncfModel.forward(input).toTensor[T]
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = ncfModel.updateGradInput(input, gradOutput).toTensor[T]
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    ncfModel.accGradParameters(input, gradOutput)
  }

  override def backward(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = ncfModel.backward(input, gradOutput).toTensor[T]
    gradInput
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (embeddingModel.parameters()._1 ++ ncfLayers.parameters()._1,
      embeddingModel.parameters()._2 ++ ncfLayers.parameters()._2)
  }

//  var embeddingModel: ConcatTable[T] = _
//  var ncfLayers: Sequential[T] = _
//  var ncfModel: Sequential[T] = _
//
//  def buildModel(): this.type = {
//    embeddingModel = ConcatTable[Tensor[T], T]()
//    val mlpEmbedding = Sequential[T]()
//    val mlpUserTable = LookupTable[T](userCount, userEmbed)
//    val mlpItemTable = LookupTable[T](itemCount, itemEmbed)
//    mlpUserTable.setWeightsBias(Array(Tensor[T](userCount, userEmbed).randn(0, 0.1)))
//    mlpItemTable.setWeightsBias(Array(Tensor[T](itemCount, itemEmbed).randn(0, 0.1)))
//    mlpEmbedding.add(ConcatTable[Tensor[T], T]()
//      .add(Sequential[T]().add(Select(2, 1)).add(mlpUserTable))
//      .add(Sequential[T]().add(Select(2, 2)).add(mlpItemTable)))
//      .add(JoinTable(2, 2))
//    embeddingModel.add(mlpEmbedding)
//
//    if (includeMF) {
//      val mfUserTable = LookupTable[T](userCount, mfEmbed)
//      val mfItemTable = LookupTable[T](itemCount, mfEmbed)
//      mfUserTable.setWeightsBias(Array(Tensor[T](userCount, mfEmbed).randn(0, 0.1)))
//      mfItemTable.setWeightsBias(Array(Tensor[T](itemCount, mfEmbed).randn(0, 0.1)))
//      val mfEmbedding = Sequential[T]()
//      mfEmbedding.add(ConcatTable[Tensor[T], T]()
//        .add(Sequential[T]().add(Select(2, 1)).add(mfUserTable))
//        .add(Sequential[T]().add(Select(2, 2)).add(mfItemTable)))
//        .add(CMulTable())
//      embeddingModel.add(mfEmbedding)
//    }
//
//    val mlpLinears = Sequential[T]()
//    val linear1 = Linear[T](itemEmbed + userEmbed, hiddenLayers(0))
//    mlpLinears.add(linear1).add(ReLU())
//    for (i <- 1 to hiddenLayers.length - 1) {
//      mlpLinears.add(Linear(hiddenLayers(i - 1), hiddenLayers(i))).add(ReLU())
//    }
//
//
//    ncfLayers = Sequential[T]()
//    if (includeMF) {
//      ncfLayers.add(ParallelTable[T]()
//        .add(mlpLinears)
//        .add(Identity[T]()))
//        .add(JoinTable(2, 2))
//        .add(Linear(mfEmbed + hiddenLayers.last, numClasses))
//    } else {
//      ncfLayers.add(Linear(hiddenLayers.last, numClasses))
//    }
//    ncfLayers.add(Sigmoid())
//
//    val ncfModel = Sequential[T]()
//
//    ncfModel.add(embeddingModel).add(ncfLayers)
//
//    this
//  }

  var embeddingModel: Graph[T] = _
  var ncfLayers: Graph[T] = _
  var ncfModel: Sequential[T] = _

  def buildModel(): this.type = {
//    embeddingModel = ConcatTable[Tensor[T], T]()
    val input = Identity().inputs()
    val userId = Select(2, 1).setName("userId").inputs(input)
    val itemId = Select(2, 2).setName("itemId").inputs(input)
    val mlpUserTable = LookupTable[T](userCount, userEmbed)
      .setName("mlpUserEmbedding")
      .setWeightsBias(Array(Tensor[T](userCount, userEmbed).randn(0, 0.1)))
      .inputs(userId)
    val mlpItemTable = LookupTable[T](itemCount, itemEmbed)
      .setName("mlpItemEmbedding")
      .setWeightsBias(Array(Tensor[T](itemCount, itemEmbed).randn(0, 0.1)))
      .inputs(itemId)
//    val mlpEmbedding = JoinTable(2, 2).inputs(mlpUserTable, mlpItemTable)
//    embeddingModel.add(mlpEmbedding)

      val mfUserTable = LookupTable[T](userCount, mfEmbed)
        .setName("mfUserEmbedding")
        .setWeightsBias(Array(Tensor[T](userCount, mfEmbed).randn(0, 0.1)))
        .inputs(userId)
      val mfItemTable = LookupTable[T](itemCount, mfEmbed)
        .setName("mfItemEmbedding")
        .setWeightsBias(Array(Tensor[T](itemCount, mfEmbed).randn(0, 0.1)))
        .inputs(itemId)
    embeddingModel =
      Graph(input, Array(mlpUserTable, mlpItemTable, mfUserTable, mfItemTable))

    val mlpUser = Identity().inputs()
    val mlpItem = Identity().inputs()
    val mfUser = Identity().inputs()
    val mfItem = Identity().inputs()

    val mlpMerge = JoinTable(2, 2).inputs(mlpUser, mlpItem)
    val mfMerge = CMulTable().inputs(mfUser, mfItem)

    var linear = Linear[T](itemEmbed + userEmbed, hiddenLayers(0)).inputs(mlpMerge)
    var relu = ReLU[T]().inputs(linear)
    for (i <- 1 to hiddenLayers.length - 1) {
      linear = Linear(hiddenLayers(i - 1), hiddenLayers(i)).inputs(relu)
      relu = ReLU().inputs(linear)
    }

    val merge = JoinTable(2, 2).inputs(mfMerge, relu)
    val finalLinear = Linear(mfEmbed + hiddenLayers.last, numClasses).inputs(merge)
    val sigmoid = Sigmoid().inputs(finalLinear)

    ncfLayers = Graph(Array(mlpUser, mlpItem, mfUser, mfItem), sigmoid)

    ncfModel = Sequential[T]()

    ncfModel.add(embeddingModel).add(ncfLayers)

    this
  }
}

object NeuralCFV2 {

  def apply[@specialized(Float, Double) T: ClassTag]
  (userCount: Int,
   itemCount: Int,
   numClasses: Int,
   userEmbed: Int,
   itemEmbed: Int,
   hiddenLayers: Array[Int],
   includeMF: Boolean = true,
   mfEmbed: Int = 20
  )(implicit ev: TensorNumeric[T]): NeuralCFV2[T] = {
    new NeuralCFV2[T](
      userCount, itemCount, numClasses, userEmbed, itemEmbed, hiddenLayers, includeMF, mfEmbed)
      .buildModel()
  }


//  def loadModel[T: ClassTag](path: String,
//                             weightPath: String = null)(implicit ev: TensorNumeric[T]):
//  NeuralCF[T] = {
//    Model.load(path, weightPath).asInstanceOf[NeuralCF[T]]
//  }
}

