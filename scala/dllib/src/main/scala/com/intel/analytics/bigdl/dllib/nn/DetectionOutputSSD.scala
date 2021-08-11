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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.BinaryTreeLSTM.{apply => _}
import com.intel.analytics.bigdl.nn.Reshape.{apply => _, createBigDLModule => _, createSerializeBigDLModule => _, getClass => _}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.{Module => _, _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.util.BboxUtil
import com.intel.analytics.bigdl.utils.{Shape, Table}
import org.apache.log4j.Logger
import DetectionOutputSSD.logger

import scala.reflect.ClassTag

/**
 * Layer to Post-process SSD output
 * @param nClasses number of classes
 * @param shareLocation whether to share location, default is true
 * @param bgLabel background label
 * @param nmsThresh nms threshold
 * @param nmsTopk nms topk
 * @param keepTopK result topk
 * @param confThresh confidence threshold
 * @param varianceEncodedInTarget if variance is encoded in target,
 *                                we simply need to retore the offset predictions,
 *                                else if variance is encoded in bbox,
 *                                we need to scale the offset accordingly.
 * @param confPostProcess whether add some additional post process to confidence prediction
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now
 */
@SerialVersionUID(5253792953255433914L)
class DetectionOutputSSD[T: ClassTag](val nClasses: Int = 21,
  val shareLocation: Boolean = true,
  val bgLabel: Int = 0,
  val nmsThresh: Float = 0.45f,
  val nmsTopk: Int = 400,
  var keepTopK: Int = 200,
  val confThresh: Float = 0.01f,
  val varianceEncodedInTarget: Boolean = false,
  val confPostProcess: Boolean = true)
  (implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Activity, T] {
  @transient private var nms: Nms = _

  def setTopK(topK: Int): this.type = {
    keepTopK = topK
    this
  }

  private def filterBboxes(decodedBboxes: Array[Tensor[Float]],
    confScores: Array[Tensor[Float]], indices: Array[Array[Int]],
    indicesNum: Array[Int]): Int = {
    var numDet = 0
    var c = 0
    while (c < nClasses) {
      if (c != bgLabel) {
        val scores = confScores(c)
        if (scores.nElement() == 0) {
          logger.warn(s"Could not find confidence predictions for label $c")
        }
        val label = if (shareLocation) decodedBboxes.length - 1 else c
        val bboxes = decodedBboxes(label)
        if (bboxes == null || bboxes.nElement() == 0) {
          logger.warn(s"Could not find locƒation predictions for label $label")
          return 0
        }
        indicesNum(c) = nms.nmsFast(scores, bboxes, nmsThresh,
          confThresh, indices(c), nmsTopk, normalized = true)

        numDet += indicesNum(c)
      }
      c += 1
    }
    if (keepTopK > -1 && numDet > keepTopK) {
      val scoreClassIndex = new Array[(Float, Int, Int)](numDet)
      var c = 0
      var count = 0
      while (c < indices.length) {
        var j = 0
        while (j < indicesNum(c)) {
          val idx = indices(c)(j)
          scoreClassIndex(count) = (confScores(c).valueAt(idx), c, idx)
          count += 1
          j += 1
        }
        indicesNum(c) = 0
        c += 1
      }
      // keep top k results per image
      val sortedPairs = scoreClassIndex.sortBy(x => -x._1)
      var i = 0
      while (i < keepTopK) {
        val label = sortedPairs(i)._2
        val idx = sortedPairs(i)._3
        indices(label)(indicesNum(label)) = idx
        indicesNum(label) += 1
        i += 1
      }
      keepTopK
    } else {
      numDet
    }
  }

  @transient private var allLocPreds: Array[Array[Tensor[Float]]] = _
  @transient private var allConfScores: Array[Array[Tensor[Float]]] = _
  @transient private var allIndices: Array[Array[Array[Int]]] = _
  @transient private var allIndicesNum: Array[Array[Int]] = _

  private def init(batch: Int, numLocClasses: Int, nPriors: Int): Unit = {
    var i = 0
    if (allLocPreds == null || allLocPreds.length < batch) {
      // the outer array is the batch, each img contains an array of results, grouped by class
      allLocPreds = new Array[Array[Tensor[Float]]](batch)
      allConfScores = new Array[Array[Tensor[Float]]](batch)
      allIndices = new Array[Array[Array[Int]]](batch)
      allIndicesNum = new Array[Array[Int]](batch)
      i = 0
      while (i < batch) {
        allLocPreds(i) = new Array[Tensor[Float]](numLocClasses)
        allConfScores(i) = new Array[Tensor[Float]](nClasses)
        allIndices(i) = new Array[Array[Int]](nClasses)
        allIndicesNum(i) = new Array[Int](nClasses)
        var c = 0
        while (c < numLocClasses) {
          allLocPreds(i)(c) = Tensor[Float](nPriors, 4)
          c += 1
        }
        c = 0
        while (c < nClasses) {
          allConfScores(i)(c) = Tensor[Float](nPriors)
          if (c != bgLabel) allIndices(i)(c) = new Array[Int](nPriors)
          c += 1
        }
        i += 1
      }

    } else {
      i = 0
      while (i < batch) {
        var c = 0
        while (c < numLocClasses) {
          allLocPreds(i)(c).resize(nPriors, 4)
          c += 1
        }
        c = 0
        while (c < nClasses) {
          allConfScores(i)(c).resize(nPriors)
          if (c != bgLabel && allIndices(i)(c).length < nPriors) {
            allIndices(i)(c) = new Array[Int](nPriors)
          }
          c += 1
        }
        i += 1
      }
    }
  }


  private val confPost = if (confPostProcess) {
    Sequential[T]()
      .add(InferReshape[T](Array(0, -1, nClasses)).setName("mbox_conf_reshape"))
      .add(TimeDistributed[T](SoftMax[T]()).setName("mbox_conf_softmax"))
      .add(InferReshape[T](Array(0, -1)).setName("mbox_conf_flatten"))
  } else {
    null
  }

  override def updateOutput(input: Table): Activity = {
    if (isTraining()) {
      output = input
      return output
    }
    if (nms == null) nms = new Nms()
    val loc = input[Tensor[Float]](1)
    val conf = if (confPostProcess) {
      confPost.forward(input[Tensor[Float]](2)).toTensor[Float]
    } else {
      input[Tensor[Float]](2)
    }
    val prior = input[Tensor[Float]](3)
    val batch = loc.size(1)
    val numLocClasses = if (shareLocation) 1 else nClasses
    val nPriors = prior.size(3) / 4

    var i = 0

    init(batch, numLocClasses, nPriors)

    BboxUtil.getLocPredictions(loc, nPriors, numLocClasses, shareLocation,
      allLocPreds)

    BboxUtil.getConfidenceScores(conf, nPriors, nClasses, allConfScores)
    val (priorBoxes, priorVariances) = BboxUtil.getPriorBboxes(prior, nPriors)

    val allDecodedBboxes = BboxUtil.decodeBboxesAll(allLocPreds, priorBoxes, priorVariances,
      numLocClasses, bgLabel, false, varianceEncodedInTarget, shareLocation,
      allLocPreds)
    val numKepts = new Array[Int](batch)
    var maxDetection = 0

    i = 0
    while (i < batch) {
      val num = filterBboxes(allDecodedBboxes(i), allConfScores(i),
        allIndices(i), allIndicesNum(i))
      numKepts(i) = num
      maxDetection = Math.max(maxDetection, num)
      i += 1
    }
    // the first element is the number of detection numbers
    val out = Tensor[Float](batch, 1 + maxDetection * 6)
    if (numKepts.sum > 0) {
      i = 0
      while (i < batch) {
        val outi = out(i + 1)
        var c = 0
        outi.setValue(1, numKepts(i))
        var offset = 2
        while (c < allIndices(i).length) {
          val indices = allIndices(i)(c)
          if (indices != null) {
            val indicesNum = allIndicesNum(i)(c)
            val locLabel = if (shareLocation) allDecodedBboxes(i).length - 1 else c
            val bboxes = allDecodedBboxes(i)(locLabel)
            var j = 0
            while (j < indicesNum) {
              val idx = indices(j)
              outi.setValue(offset, c)
              outi.setValue(offset + 1, allConfScores(i)(c).valueAt(idx))
              outi.setValue(offset + 2, bboxes.valueAt(idx, 1))
              outi.setValue(offset + 3, bboxes.valueAt(idx, 2))
              outi.setValue(offset + 4, bboxes.valueAt(idx, 3))
              outi.setValue(offset + 5, bboxes.valueAt(idx, 4))
              offset += 6
              j += 1
            }
          }
          c += 1
        }
        i += 1
      }
    }
    output = out
    output
  }

  override def updateGradInput(input: Table, gradOutput: Activity): Table = {
    gradInput = gradOutput.toTable
    gradInput
  }

  override def clearState(): DetectionOutputSSD.this.type = {
    nms = null
    allLocPreds = null
    allConfScores = null
    allIndices = null
    allIndicesNum = null
    if (null != confPost) confPost.clearState()
    this
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    if (isTraining()) {
      return inputShape
    }
    throw new RuntimeException("Not support computeOutputShape for DetectionOutputSSD Inference")
  }
}

object DetectionOutputSSD {
  val logger = Logger.getLogger(getClass)

  def apply[@specialized(Float) T: ClassTag]
  (param: DetectionOutputParam, postProcess: Boolean = true)
    (implicit ev: TensorNumeric[T]): DetectionOutputSSD[T] =
    new DetectionOutputSSD[T](param.nClasses,
      param.shareLocation,
      param.bgLabel,
      param.nmsThresh,
      param.nmsTopk,
      param.keepTopK,
      param.confThresh,
      param.varianceEncodedInTarget,
      postProcess)
}


case class DetectionOutputParam(nClasses: Int = 21, shareLocation: Boolean = true, bgLabel: Int = 0,
  nmsThresh: Float = 0.45f, nmsTopk: Int = 400, var keepTopK: Int = 200,
  confThresh: Float = 0.01f,
  varianceEncodedInTarget: Boolean = false)
