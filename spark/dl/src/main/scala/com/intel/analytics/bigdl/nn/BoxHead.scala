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

import breeze.linalg.dim
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.bigdl.transform.vision.image.util.BboxUtil
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable.ArrayBuffer

class BoxHead(
  val inChannels: Int = 0,
  val resolution: Int = 0,
  val scales: Array[Float],
  val samplingRatio: Float = 2.0f,
  val scoreThresh: Float = 0.05f,
  val nmsThresh: Float = 0.5f,
  val maxPerImage: Int = 100,
  val outputSize: Int = 1024,
  val numClasses: Int = 81 // coco dataset class number
  )(implicit ev: TensorNumeric[Float])
  extends BaseModule[Float] {

  override def buildModel(): Module[Float] = {
    val featureExtractor = this.featureExtractor(
      inChannels, resolution, scales, samplingRatio.toInt, outputSize)

    val clsPre = this.clsPredictor(numClasses, outputSize)
    val bboxPre = this.bboxPredictor(numClasses, outputSize)

    val postProcessor = new BoxPostProcessor(scoreThresh, nmsThresh, maxPerImage, numClasses)

    val features = Input()
    val proposals = Input()

    val boxFeatures = featureExtractor.inputs(features, proposals)
    val classLogits = clsPre.inputs(boxFeatures)
    val boxRegression = bboxPre.inputs(boxFeatures)
    val result = postProcessor.inputs(classLogits, boxRegression, proposals)

    Graph(Array(features, proposals), Array(boxFeatures, result))
  }

  private[nn] def clsPredictor(numClass: Int,
                               inChannels: Int): Module[Float] = {
    val cls_score = Linear[Float](inChannels, numClass)
    cls_score.weight.apply1(_ => RNG.normal(0, 0.01).toFloat)
    cls_score.bias.fill(0.0f)
    cls_score.asInstanceOf[Module[Float]]
  }

  private[nn] def bboxPredictor(numClass: Int,
                               inChannels: Int): Module[Float] = {
    val bbox_pred = Linear[Float](inChannels, numClass * 4)
    bbox_pred.weight.apply1(_ => RNG.normal(0, 0.001).toFloat)
    bbox_pred.bias.fill(0.0f)
    bbox_pred.asInstanceOf[Module[Float]]
  }

  private[nn] def featureExtractor(inChannels: Int,
                                   resolution: Int,
                                   scales: Array[Float], samplingRatio: Int,
                                   representationSize: Int): Module[Float] = {
    val pooler = new Pooler(resolution, scales, samplingRatio)
    val inputSize = inChannels * math.pow(resolution, 2).toInt

    val fc1 = Linear[Float](inputSize, representationSize, withBias = true)
      .setInitMethod(Xavier, Zeros)
    val fc2 = Linear[Float](representationSize, representationSize, withBias = true)
      .setInitMethod(Xavier, Zeros)

    val model = Sequential[Float]()
      .add(pooler)
      .add(InferReshape(Array(0, -1)))
      .add(fc1)
      .add(ReLU[Float]())
      .add(fc2)
      .add(ReLU[Float]())

    model
  }
}

private[nn] class BoxPostProcessor(
    scoreThresh: Float,
    nmsThresh: Float,
    maxPerImage: Int,
    nClasses: Int,
    weight: Array[Float] = Array(1.0f, 1.0f, 1.0f, 1.0f)
  ) (implicit ev: TensorNumeric[Float]) extends AbstractModule[Table, Activity, Float] {

  private val softMax = SoftMax[Float]()
  private val nmsTool: Nms = new Nms
  @transient var boxesBuf: Tensor[Float] = _

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
      clsInd += 1
    }
    // Limit to max_per_image detections *over all classes*
    if (maxPerImage > 0) {
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
  override def updateOutput(input: Table): Activity = {
    if (isTraining()) {
      output = input
      return output
    }
    val classLogits = input[Tensor[Float]](1)
    val boxRegression = input[Tensor[Float]](2)
    val bbox = input[Tensor[Float]](3)

    if (boxesBuf == null) boxesBuf = Tensor[Float]
    boxesBuf.resizeAs(boxRegression)

    val class_prob = softMax.forward(classLogits)
    BboxUtil.decodeWithWeight(boxRegression, bbox, weight, boxesBuf)

    val boxesInImage = bbox.size(1)
    val proposals_split = boxesBuf.split(boxesInImage, dim = 1)
    val class_prob_split = class_prob.split(boxesInImage, dim = 1)

    val roilabels = filterResults(proposals_split(0), class_prob_split(0), nClasses)
    output = resultToTensor(roilabels)
    output
  }

  override def updateGradInput(input: Table, gradOutput: Activity): Table = {
    gradInput = gradOutput.toTable
    gradInput
  }
}

object BoxHead {
  def apply(inChannels: Int = 0,
  resolution: Int = 0,
  scales: Array[Float],
  samplingRatio: Float = 2.0f,
  scoreThresh: Float = 0.05f,
  nmsThresh: Float = 0.5f,
  maxPerImage: Int = 100,
  outputSize: Int = 1024,
  numClasses: Int = 81 // coco dataset class number
  ) ( implicit ev: TensorNumeric[Float]): BoxHead =
    new BoxHead(inChannels, resolution, scales, samplingRatio,
      scoreThresh, nmsThresh, maxPerImage, outputSize)
}
