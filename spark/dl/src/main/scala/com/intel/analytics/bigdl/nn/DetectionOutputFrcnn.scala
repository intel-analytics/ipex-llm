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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.bigdl.transform.vision.image.util.BboxUtil
import com.intel.analytics.bigdl.utils.Table
import org.apache.log4j.Logger

import scala.collection.mutable.ArrayBuffer


object DetectionOutputFrcnn {
  val logger = Logger.getLogger(this.getClass)

  def apply(nmsThresh: Float = 0.3f, nClasses: Int = 21,
  bboxVote: Boolean = false, maxPerImage: Int = 100, thresh: Double = 0.05)(
    implicit ev: TensorNumeric[Float]): DetectionOutputFrcnn =
    new DetectionOutputFrcnn(nmsThresh, nClasses, bboxVote, maxPerImage, thresh)
}

/**
 * Post process Faster-RCNN models
 * @param nmsThresh nms threshold
 * @param nClasses number of classes
 * @param bboxVote whether to vote for detections
 * @param maxPerImage limit max number of detections per image
 * @param thresh score threshold
 */
@SerialVersionUID(5253792953255433914L)
class DetectionOutputFrcnn(var nmsThresh: Float = 0.3f, val nClasses: Int = 21,
  var bboxVote: Boolean = false, var maxPerImage: Int = 100, var thresh: Double = 0.05)(
  implicit ev: TensorNumeric[Float]) extends AbstractModule[Table, Activity, Float] {

  @transient var nmsTool: Nms = _

  // scores (N, clsNum)
  // boxes (N, 4 * clsNum)
  private def postProcess(scores: Tensor[Float], boxes: Tensor[Float])
  : Array[RoiLabel] = {
    require(scores.size(1) == boxes.size(1))
    val results = new Array[RoiLabel](nClasses)
    // skip j = 0, because it's the background class
    var clsInd = 1
    while (clsInd < nClasses) {
      results(clsInd) = postProcessOneClass(scores, boxes, clsInd)
      clsInd += 1
    }

    // Limit to max_per_image detections *over all classes*
    if (maxPerImage > 0) {
      limitMaxPerImage(results)
    }
    results
  }

  private def resultToTensor(results: Array[RoiLabel]): Tensor[Float] = {
    var maxDetection = 0
    results.foreach(res => {
      if (null != res) {
        maxDetection += res.size()
      }
    })
    val out = Tensor[Float](1, 1 + maxDetection * 6)
    val outi = out(1)

    outi.setValue(1, maxDetection)
    var offset = 2
    (0 until nClasses).foreach(c => {
      val label = results(c)
      if (null != label) {
        (1 to label.size()).foreach(j => {
          outi.setValue(offset, c)
          outi.setValue(offset + 1, label.classes.valueAt(j))
          outi.setValue(offset + 2, label.bboxes.valueAt(j, 1))
          outi.setValue(offset + 3, label.bboxes.valueAt(j, 2))
          outi.setValue(offset + 4, label.bboxes.valueAt(j, 3))
          outi.setValue(offset + 5, label.bboxes.valueAt(j, 4))
          offset += 6
        })
      }
    })
    out
  }

  @transient private var areas: Tensor[Float] = _

  private def postProcessOneClass(scores: Tensor[Float], boxes: Tensor[Float],
    clsInd: Int): RoiLabel = {
    val inds = (1 to scores.size(1)).filter(ind =>
      scores.valueAt(ind, clsInd + 1) > thresh).toArray
    if (inds.length == 0) return null
    val clsScores = selectTensor(scores.select(2, clsInd + 1), inds, 1)
    val clsBoxes = selectTensor(boxes.narrow(2, clsInd * 4 + 1, 4), inds, 1)

    val keepN = nmsTool.nms(clsScores, clsBoxes, nmsThresh, inds)

    val bboxNms = selectTensor(clsBoxes, inds, 1, keepN)
    val scoresNms = selectTensor(clsScores, inds, 1, keepN)
    if (bboxVote) {
      if (areas == null) areas = Tensor[Float]
      BboxUtil.bboxVote(scoresNms, bboxNms, clsScores, clsBoxes, areas)
    } else {
      RoiLabel(scoresNms, bboxNms)
    }
  }

  private def selectTensor(matrix: Tensor[Float], indices: Array[Int],
    dim: Int, indiceLen: Int = -1, out: Tensor[Float] = null): Tensor[Float] = {
    assert(dim == 1 || dim == 2)
    var i = 1
    val n = if (indiceLen == -1) indices.length else indiceLen
    if (matrix.nDimension() == 1) {
      val res = if (out == null) {
        Tensor[Float](n)
      } else {
        out.resize(n)
      }
      while (i <= n) {
        res.update(i, matrix.valueAt(indices(i - 1)))
        i += 1
      }
      return res
    }
    // select rows
    if (dim == 1) {
      val res = if (out == null) {
        Tensor[Float](n, matrix.size(2))
      } else {
        out.resize(n, matrix.size(2))
      }
      while (i <= n) {
        res.update(i, matrix(indices(i - 1)))
        i += 1
      }
      res
    } else {
      val res = if (out == null) {
        Tensor[Float](matrix.size(1), n)
      } else {
        out.resize(matrix.size(1), n)
      }
      while (i <= n) {
        var rid = 1
        val value = matrix.select(2, indices(i - 1))
        while (rid <= res.size(1)) {
          res.setValue(rid, i, value.valueAt(rid))
          rid += 1
        }
        i += 1
      }
      res
    }
  }

  def limitMaxPerImage(results: Array[RoiLabel]): Unit = {
    val nImageScores = (1 until nClasses).map(j => if (results(j) == null) 0
    else results(j).classes.size(1)).sum
    if (nImageScores > maxPerImage) {
      val imageScores = ArrayBuffer[Float]()
      var j = 1
      while (j < nClasses) {
        if (results(j) != null) {
          val res = results(j).classes
          if (res.nElement() > 0) {
            res.apply1(x => {
              imageScores.append(x)
              x
            })
          }
        }
        j += 1
      }
      val imageThresh = imageScores.sortWith(_ < _)(imageScores.length - maxPerImage)
      j = 1
      while (j < nClasses) {
        if (results(j) != null) {
          val box = results(j).bboxes
          val keep = (1 to box.size(1)).filter(x =>
            box.valueAt(x, box.size(2)) >= imageThresh).toArray
          val selectedScores = selectTensor(results(j).classes, keep, 1)
          val selectedBoxes = selectTensor(results(j).bboxes, keep, 1)
          if (selectedScores.nElement() == 0) {
            results(j).classes.set()
            results(j).bboxes.set()
          } else {
            results(j).classes.resizeAs(selectedScores).copy(selectedScores)
            results(j).bboxes.resizeAs(selectedBoxes).copy(selectedBoxes)
          }
        }
        j += 1
      }
    }
  }

  @transient var boxesBuf: Tensor[Float] = _

  def process(scores: Tensor[Float],
    boxDeltas: Tensor[Float],
    rois: Tensor[Float],
    imInfo: Tensor[Float]): Array[RoiLabel] = {
    if (nmsTool == null) nmsTool = new Nms
    // post process
    // unscale back to raw image space
    if (boxesBuf == null) boxesBuf = Tensor[Float]
    boxesBuf.resize(rois.size(1), 4).copy(rois.narrow(2, 2, 4))
    BboxUtil.scaleBBox(boxesBuf, 1 / imInfo.valueAt(1, 3), 1 / imInfo.valueAt(1, 4))
    // Apply bounding-box regression deltas
    val predBoxes = BboxUtil.bboxTransformInv(boxesBuf, boxDeltas)
    BboxUtil.clipBoxes(predBoxes, imInfo.valueAt(1, 1) / imInfo.valueAt(1, 3),
      imInfo.valueAt(1, 2) / imInfo.valueAt(1, 4))
    val res = postProcess(scores, predBoxes)
    res
  }

  override def updateOutput(input: Table): Activity = {
    if (isTraining()) {
      output = input
      return output
    }
    val imInfo = input[Tensor[Float]](1)
    val roisData = input[Activity](2)
    val rois = if (roisData.isTable) roisData.toTable[Tensor[Float]](1)
    else roisData.toTensor[Float]
    val boxDeltas = input[Tensor[Float]](3)
    val scores = input[Tensor[Float]](4)
    require(imInfo.dim() == 2 && imInfo.size(1) == 1 && imInfo.size(2) == 4,
      s"imInfo should be a 1x4 tensor, while actual is ${imInfo.size().mkString("x")}")
    require(rois.size(2) == 5,
      s"rois is a Nx5 tensor, while actual is ${rois.size().mkString("x")}")
    require(boxDeltas.size(2) == nClasses * 4,
      s"boxDeltas is a Nx(nClasses * 4) tensor, while actual is ${boxDeltas.size().mkString("x")}")
    require(scores.size(2) == nClasses,
      s"scores is a NxnClasses tensor, while actual is ${scores.size().mkString("x")}")
    output = resultToTensor(process(scores, boxDeltas, rois, imInfo))
    output
  }

  override def updateGradInput(input: Table, gradOutput: Activity): Table = {
    gradInput = gradOutput.toTable
    gradInput
  }
}
