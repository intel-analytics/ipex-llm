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

import breeze.linalg.{*, dim}
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.bigdl.transform.vision.image.util.BboxUtil
import com.intel.analytics.bigdl.utils.{T, Table}
import com.sun.org.apache.xpath.internal.operations.Bool
import org.apache.spark.sql.catalyst.expressions.If
import org.dmg.pmml.{Apply, False}

import scala.collection.mutable.ArrayBuffer

class FPNPostProcessor(
    scoreThresh: Float,
    nmsThresh: Float,
    detections_per_img: Int
  ) (implicit ev: TensorNumeric[Float]) extends AbstractModule[Table, Tensor[Float], Float] {

  private val softMax = SoftMax[Float]()
  private val weight = Tensor[Float](T(1.0f, 1.0f, 1.0f, 1.0f))
  private val nmsTool: Nms = new Nms
  private var areas: Tensor[Float] = _
  private var nClasses: Int = _

  /**
   * Returns bounding-box detection results by thresholding on scores and
   * applying non-maximum suppression (NMS).
   */
  private[nn] def filterResults(boxes: Tensor[Float], scores: Tensor[Float],
                                num_classes: Int): Array[RoiLabel] = {
    val dim = num_classes * 4
    boxes.resize(Array(boxes.nElement() / dim, dim))
    scores.resize(Array(scores.nElement() / num_classes, num_classes))

    val results = new Array[RoiLabel](num_classes)
    var clsInd = 1
    while (clsInd < num_classes) {
      results(clsInd) = postProcessOneClass(scores, boxes, clsInd)
      if (results(clsInd) == null) {
        println(clsInd)
      }
      clsInd += 1
    }
    // Limit to max_per_image detections *over all classes*
    if (detections_per_img > 0) {
      limitMaxPerImage(results)
    }
    results
  }

  private def postProcessOneClass(scores: Tensor[Float], boxes: Tensor[Float],
                          clsInd: Int): RoiLabel = {
    val inds = (1 to scores.size(1)).filter(ind =>
      scores.valueAt(ind, clsInd + 1) > scoreThresh).toArray
    if (inds.length == 0) return null
    val clsScores = selectTensor(scores.select(2, clsInd + 1), inds, 1)
    val clsBoxes = selectTensor(boxes.narrow(2, clsInd * 4 + 1, 4), inds, 1)

    val keepN = nmsTool.nms(clsScores, clsBoxes, nmsThresh, inds)

    val bboxNms = selectTensor(clsBoxes, inds, 1, keepN)
    val scoresNms = selectTensor(clsScores, inds, 1, keepN)

    RoiLabel(scoresNms, bboxNms)
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

  private def limitMaxPerImage(results: Array[RoiLabel]): Unit = {
    val nImageScores = (1 until nClasses).map(j => if (results(j) == null) 0
    else results(j).classes.size(1)).sum
    if (nImageScores > detections_per_img) {
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
      val imageThresh = imageScores.sortWith(_ < _)(imageScores.length - detections_per_img)
      j = 1
      while (j < nClasses) {
        if (results(j) != null) {
          val box = results(j).classes
          val keep = (1 to box.size(1)).filter(x =>
            box.valueAt(x) >= imageThresh).toArray
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

  /**
    * input contains:the class logits, the box_regression and
    * bounding boxes that are used as reference, one for ech image
    * @param input
    * @return boxlist contains labels and scores
    */
  override def updateOutput(input: Table): Tensor[Float] = {
    val inputFlatten = input.flatten()
    val class_logits = inputFlatten[Tensor[Float]](1)
    val box_regression = inputFlatten[Tensor[Float]](2)
    val bbox = inputFlatten[Tensor[Float]](3)

    val class_prob = softMax.forward(class_logits)
    val proposals = Tensor[Float]().resizeAs(box_regression).copy(box_regression)

    BboxUtil.decodeWithWeight(box_regression, bbox, weight, proposals)

    nClasses = class_prob.size(2)
    val boxes_per_image = bbox.size(1)

    val proposals_split = proposals.split(boxes_per_image, dim = 1)
    val class_prob_split = class_prob.split(boxes_per_image, dim = 1)

    val roilabels = filterResults(proposals_split(0), class_prob_split(0), nClasses)
    output = resultToTensor(roilabels)
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[Float]): Table = {
    throw new UnsupportedOperationException("FPNPostProcessor only support inference")
  }
}
