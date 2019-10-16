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
  val inChannels: Int,
  val resolution: Int,
  val scales: Array[Float],
  val samplingRatio: Int,
  val scoreThresh: Float,
  val nmsThresh: Float,
  val maxPerImage: Int,
  val outputSize: Int,
  val numClasses: Int
  )(implicit ev: TensorNumeric[Float])
  extends BaseModule[Float] {

  override def buildModel(): Module[Float] = {
    val featureExtractor = this.featureExtractor(
      inChannels, resolution, scales, samplingRatio, outputSize)

    val clsPredictor = this.clsPredictor(numClasses, outputSize)
    val bboxPredictor = this.bboxPredictor(numClasses, outputSize)

    val weight = Array(10.0f, 10.0f, 5.0f, 5.0f)
    val postProcessor = new BoxPostProcessor(scoreThresh, nmsThresh,
      maxPerImage, numClasses, weight = weight)

    val features = Input()
    val proposals = Input()

    val boxFeatures = featureExtractor.inputs(features, proposals)
    val classLogits = clsPredictor.inputs(boxFeatures)
    val boxRegression = bboxPredictor.inputs(boxFeatures)
    val result = postProcessor.inputs(classLogits, boxRegression, proposals)

    Graph(Array(features, proposals), Array(boxFeatures, result))
  }

  private[nn] def clsPredictor(numClass: Int,
                               inChannels: Int): Module[Float] = {
    val clsScore = Linear[Float](inChannels, numClass)
    clsScore.weight.apply1(_ => RNG.normal(0, 0.01).toFloat)
    clsScore.bias.fill(0.0f)
    clsScore.asInstanceOf[Module[Float]]
  }

  private[nn] def bboxPredictor(numClass: Int,
                               inChannels: Int): Module[Float] = {
    val bboxRegression = Linear[Float](inChannels, numClass * 4)
    bboxRegression.weight.apply1(_ => RNG.normal(0, 0.001).toFloat)
    bboxRegression.bias.fill(0.0f)
    bboxRegression.asInstanceOf[Module[Float]]
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
    val scoreThresh: Float,
    val nmsThresh: Float,
    val maxPerImage: Int,
    val nClasses: Int,
    val weight: Array[Float] = Array(10.0f, 10.0f, 5.0f, 5.0f)
  ) (implicit ev: TensorNumeric[Float]) extends AbstractModule[Table, Table, Float] {

  private val softMax = SoftMax[Float]()
  private val nmsTool: Nms = new Nms
  @transient  private var boxesBuf: Tensor[Float] = null

  /**
   * Returns bounding-box detection results by thresholding on scores and
   * applying non-maximum suppression (NMS).
   */
  private[nn] def filterResults(boxes: Tensor[Float], scores: Tensor[Float],
                                numOfClasses: Int): Array[RoiLabel] = {
    val dim = numOfClasses * 4
    boxes.resize(Array(boxes.nElement() / dim, dim))
    scores.resize(Array(scores.nElement() / numOfClasses, numOfClasses))

    val results = new Array[RoiLabel](numOfClasses)
    // skip clsInd = 0, because it's the background class
    var clsInd = 1
    while (clsInd < numOfClasses) {
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
    require(dim == 1 || dim == 2, s"dim should be 1 or 2, but get ${dim}")
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

  private def resultToTensor(results: Array[RoiLabel], labels: Tensor[Float], bbox: Tensor[Float])
    : Unit = {
    var maxDetection = 0
    results.foreach(res => {
      if (null != res) {
        maxDetection += res.size()
      }
    })

    labels.resize(maxDetection)
    bbox.resize(maxDetection, 4)

    var offset = 1
    (0 until nClasses).foreach(c => {
      val label = results(c)
      if (null != label) {
        (1 to label.size()).foreach(j => {
          labels.setValue(offset, c)
          bbox.setValue(offset, 1, label.bboxes.valueAt(j, 1))
          bbox.setValue(offset, 2, label.bboxes.valueAt(j, 2))
          bbox.setValue(offset, 3, label.bboxes.valueAt(j, 3))
          bbox.setValue(offset, 4, label.bboxes.valueAt(j, 4))
          offset += 1
        })
      }
    })
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
   * @return labels and bbox
   */
  override def updateOutput(input: Table): Table = {
    if (isTraining()) {
      output = input
      return output
    }
    val classLogits = input[Tensor[Float]](1)
    val boxRegression = input[Tensor[Float]](2)
    val bbox = if (input(3).isInstanceOf[Tensor[Float]]) {
      input[Tensor[Float]](3)
    } else input[Table](3)[Tensor[Float]](1)

    if (boxesBuf == null) boxesBuf = Tensor[Float]
    boxesBuf.resizeAs(boxRegression)

    val classProb = softMax.forward(classLogits)
    BboxUtil.decodeWithWeight(boxRegression, bbox, weight, boxesBuf)

    val boxesInImage = bbox.size(1)
    val proposalSplit = boxesBuf.split(boxesInImage, dim = 1)
    val classProbSplit = classProb.split(boxesInImage, dim = 1)

    val roilabels = filterResults(proposalSplit(0), classProbSplit(0), nClasses)

    if (output.toTable.length() == 0) {
      output.toTable(1) = Tensor[Float]() // for labels
      output.toTable(2) = Tensor[Float]() // for bbox
    }

    resultToTensor(roilabels, output.toTable(1), output.toTable(2))
    output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    gradInput = gradOutput.toTable
    gradInput
  }
}

object BoxHead {
  def apply(inChannels: Int,
  resolution: Int = 7,
  scales: Array[Float] = Array[Float](0.25f, 0.125f, 0.0625f, 0.03125f),
  samplingRatio: Int = 2,
  scoreThresh: Float = 0.05f,
  nmsThresh: Float = 0.5f,
  maxPerImage: Int = 100,
  outputSize: Int = 1024,
  numClasses: Int = 81 // coco dataset class number
  ) ( implicit ev: TensorNumeric[Float]): BoxHead =
    new BoxHead(inChannels, resolution, scales, samplingRatio,
      scoreThresh, nmsThresh, maxPerImage, outputSize, numClasses)
}
