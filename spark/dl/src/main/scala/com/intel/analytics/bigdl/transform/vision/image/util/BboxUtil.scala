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

package com.intel.analytics.bigdl.transform.vision.image.util

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import org.apache.log4j.Logger

object BboxUtil {
  val logger = Logger.getLogger(getClass)

  def decodeRois(output: Tensor[Float]): Tensor[Float] = {
    // ignore if decoded
    if (output.nElement() < 6 || output.dim() == 2) return output
    val num = output.valueAt(1).toInt
    require(num >= 0)
    if (num == 0) {
      Tensor[Float]()
    } else {
      output.narrow(1, 2, num * 6).view(num, 6)
    }
  }

  // inplace scale
  def scaleBBox(bboxes: Tensor[Float], height: Float, width: Float): Unit = {
    if (bboxes.nElement() == 0) return
    bboxes.select(2, 1).mul(width)
    bboxes.select(2, 2).mul(height)
    bboxes.select(2, 3).mul(width)
    bboxes.select(2, 4).mul(height)
  }

  /**
   * Note that the output are stored in input deltas
   * @param boxes (N, 4)
   * @param deltas (N, 4a)
   * @return
   */
  def bboxTransformInv(boxes: Tensor[Float], deltas: Tensor[Float],
                       normalized: Boolean = false): Tensor[Float] = {
    if (boxes.size(1) == 0) {
      return boxes
    }
    val output = Tensor[Float]().resizeAs(deltas).copy(deltas)
    require(boxes.size(2) == 4,
      s"boxes size ${boxes.size().mkString(",")} do not satisfy N*4 size")
    require(output.size(2) % 4 == 0,
      s"and deltas size ${output.size().mkString(",")} do not satisfy N*4a size")
    val boxesArr = boxes.storage().array()
    var offset = boxes.storageOffset() - 1
    val rowLength = boxes.stride(1)
    val deltasArr = output.storage().array()
    var i = 0
    val repeat = output.size(2) / boxes.size(2)
    var deltasoffset = output.storageOffset() - 1
    while (i < boxes.size(1)) {
      val x1 = boxesArr(offset)
      val y1 = boxesArr(offset + 1)
      val width = if (!normalized) boxesArr(offset + 2) - x1 + 1 else boxesArr(offset + 2) - x1
      val height = if (!normalized) boxesArr(offset + 3) - y1 + 1 else boxesArr(offset + 3) - y1
      var j = 0
      while (j < repeat) {
        j += 1
        // dx1*width + centerX
        val predCtrX = deltasArr(deltasoffset) * width + x1 + width / 2
        // dy1*height + centerY
        val predCtrY = deltasArr(deltasoffset + 1) * height + y1 + height / 2
        // exp(dx2)*width/2
        val predW = Math.exp(deltasArr(deltasoffset + 2)).toFloat * width / 2
        // exp(dy2)*height/2
        val predH = Math.exp(deltasArr(deltasoffset + 3)).toFloat * height / 2
        deltasArr(deltasoffset) = predCtrX - predW
        deltasArr(deltasoffset + 1) = predCtrY - predH
        deltasArr(deltasoffset + 2) = predCtrX + predW
        deltasArr(deltasoffset + 3) = predCtrY + predH
        deltasoffset += rowLength
      }
      offset += rowLength
      i += 1
    }
    output
  }

  /**
   * Clip boxes to image boundaries.
   * set the score of all boxes with any side smaller than minSize to 0
   * @param boxes N * 4a
   * @param height height of image
   * @param width width of image
   * @param minH min height limit
   * @param minW min width limit
   * @param scores scores for boxes
   * @return the number of boxes kept (score > 0)
   */
  def clipBoxes(boxes: Tensor[Float], height: Float, width: Float, minH: Float = 0,
    minW: Float = 0, scores: Tensor[Float] = null): Int = {
    require(boxes.size(2) % 4 == 0, "boxes should have the shape N*4a")
    val boxesArr = boxes.storage().array()
    var offset = boxes.storageOffset() - 1
    val scoresArr = if (scores != null) scores.storage().array() else null
    var scoreOffset = if (scores != null) scores.storageOffset() - 1 else -1
    var i = 0
    var count = 0
    val h = height - 1
    val w = width - 1
    val repeat = boxes.size(2) / 4
    while (i < boxes.size(1)) {
      var r = 0
      while (r < repeat) {
        boxesArr(offset) = Math.max(Math.min(boxesArr(offset), w), 0)
        boxesArr(offset + 1) = Math.max(Math.min(boxesArr(offset + 1), h), 0)
        boxesArr(offset + 2) = Math.max(Math.min(boxesArr(offset + 2), w), 0)
        boxesArr(offset + 3) = Math.max(Math.min(boxesArr(offset + 3), h), 0)

        if (scores != null) {
          val width = boxesArr(offset + 2) - boxesArr(offset) + 1
          if (width < minW) {
            scoresArr(scoreOffset) = 0
          } else {
            val height = boxesArr(offset + 3) - boxesArr(offset + 1) + 1
            if (height < minH) scoresArr(scoreOffset) = 0
            else count += 1
          }
          scoreOffset += 1
        }
        r += 1
        offset += 4
      }
      i += 1
    }
    count
  }

  def getLocPredictions(loc: Tensor[Float], numPredsPerClass: Int, numClasses: Int,
    shareLocation: Boolean, locPredsBuf: Array[Array[Tensor[Float]]] = null)
  : Array[Array[Tensor[Float]]] = {
    // the outer array is the batch, each img contains an array of results, grouped by class
    val locPreds = if (locPredsBuf == null) {
      val out = new Array[Array[Tensor[Float]]](loc.size(1))
      var i = 0
      while (i < loc.size(1)) {
        out(i) = new Array[Tensor[Float]](numClasses)
        var c = 0
        while (c < numClasses) {
          out(i)(c) = Tensor[Float](numPredsPerClass, 4)
          c += 1
        }
        i += 1
      }
      out
    } else {
      locPredsBuf
    }
    var i = 0
    val locData = loc.storage().array()
    var locDataOffset = loc.storageOffset() - 1
    while (i < loc.size(1)) {
      val labelBbox = locPreds(i)
      var p = 0
      while (p < numPredsPerClass) {
        val startInd = p * numClasses * 4 + locDataOffset
        var c = 0
        while (c < numClasses) {
          val label = if (shareLocation) labelBbox.length - 1 else c
          val boxData = labelBbox(label).storage().array()
          val boxOffset = p * 4 + labelBbox(label).storageOffset() - 1
          val offset = startInd + c * 4
          boxData(boxOffset) = locData(offset)
          boxData(boxOffset + 1) = locData(offset + 1)
          boxData(boxOffset + 2) = locData(offset + 2)
          boxData(boxOffset + 3) = locData(offset + 3)
          c += 1
        }
        p += 1
      }
      locDataOffset += numPredsPerClass * numClasses * 4
      i += 1
    }
    locPreds
  }

  def getConfidenceScores(conf: Tensor[Float], numPredsPerClass: Int, numClasses: Int,
    confBuf: Array[Array[Tensor[Float]]] = null)
  : Array[Array[Tensor[Float]]] = {
    val confPreds = if (confBuf == null) {
      val out = new Array[Array[Tensor[Float]]](conf.size(1))
      var i = 0
      while (i < conf.size(1)) {
        out(i) = new Array[Tensor[Float]](numClasses)
        var c = 0
        while (c < numClasses) {
          out(i)(c) = Tensor[Float](numPredsPerClass)
          c += 1
        }
        i += 1
      }
      out
    }
    else confBuf
    val confData = conf.storage().array()
    var confDataOffset = conf.storageOffset() - 1
    var i = 0
    while (i < conf.size(1)) {
      val labelScores = confPreds(i)
      var p = 0
      while (p < numPredsPerClass) {
        val startInd = p * numClasses + confDataOffset
        var c = 0
        while (c < numClasses) {
          labelScores(c).setValue(p + 1, confData(startInd + c))
          c += 1
        }
        p += 1
      }
      confDataOffset += numPredsPerClass * numClasses
      i += 1
    }
    confPreds
  }

  def getPriorBboxes(prior: Tensor[Float], nPriors: Int): (Tensor[Float], Tensor[Float]) = {
    val array = prior.storage()
    val aOffset = prior.storageOffset()
    val priorBoxes = Tensor(array, aOffset, Array(nPriors, 4))
    val priorVariances = Tensor(array, aOffset + nPriors * 4, Array(nPriors, 4))
    (priorBoxes, priorVariances)
  }

  def decodeBboxesAll(allLocPreds: Array[Array[Tensor[Float]]], priorBoxes: Tensor[Float],
    priorVariances: Tensor[Float], nClasses: Int, bgLabel: Int, clipBoxes: Boolean,
    varianceEncodedInTarget: Boolean, shareLocation: Boolean,
    output: Array[Array[Tensor[Float]]] = null)
  : Array[Array[Tensor[Float]]] = {
    val batch = allLocPreds.length
    val allDecodeBboxes = if (output == null) {
      val all = new Array[Array[Tensor[Float]]](batch)
      var i = 0
      while (i < batch) {
        all(i) = new Array[Tensor[Float]](nClasses)
        i += 1
      }
      all
    } else {
      require(output.length == batch)
      output
    }
    var i = 0
    while (i < batch) {
      val decodedBoxes = allDecodeBboxes(i)
      var c = 0
      while (c < nClasses) {
        // Ignore background class.
        if (shareLocation || c != bgLabel) {
          // Something bad happened if there are no predictions for current label.
          if (allLocPreds(i)(c).nElement() == 0) {
            logger.warn(s"Could not find location predictions for label $c")
          }
          val labelLocPreds = allLocPreds(i)(c)
          decodedBoxes(c) = decodeBoxes(priorBoxes, priorVariances, clipBoxes,
            labelLocPreds, varianceEncodedInTarget, labelLocPreds)
        }
        c += 1
      }
      allDecodeBboxes(i) = decodedBoxes
      i += 1
    }
    allDecodeBboxes
  }

  def decodeBoxes(priorBoxes: Tensor[Float], priorVariances: Tensor[Float],
    isClipBoxes: Boolean, bboxes: Tensor[Float],
    varianceEncodedInTarget: Boolean, output: Tensor[Float] = null): Tensor[Float] = {
    require(priorBoxes.size(1) == priorVariances.size(1))
    require(priorBoxes.size(1) == bboxes.size(1))
    val numBboxes = priorBoxes.size(1)
    if (numBboxes > 0) {
      require(priorBoxes.size(2) == 4)
    }
    val decodedBboxes = if (output == null) Tensor[Float](numBboxes, 4)
    else output.resizeAs(priorBoxes)
    var i = 1
    while (i <= numBboxes) {
      decodeSingleBbox(i, priorBoxes,
        priorVariances, isClipBoxes, bboxes, varianceEncodedInTarget, decodedBboxes)
      i += 1
    }
    decodedBboxes
  }

  private def decodeSingleBbox(i: Int, priorBox: Tensor[Float], priorVariance: Tensor[Float],
    isClipBoxes: Boolean, bbox: Tensor[Float], varianceEncodedInTarget: Boolean,
    decodedBoxes: Tensor[Float]): Unit = {
    val x1 = priorBox.valueAt(i, 1)
    val y1 = priorBox.valueAt(i, 2)
    val x2 = priorBox.valueAt(i, 3)
    val y2 = priorBox.valueAt(i, 4)
    val priorWidth = x2 - x1
    require(priorWidth > 0)
    val priorHeight = y2 - y1
    require(priorHeight > 0)
    val pCenterX = (x1 + x2) / 2
    val pCenterY = (y1 + y2) / 2
    var decodeCenterX = 0f
    var decodeCenterY = 0f
    var decodeWidth = 0f
    var decodedHeight = 0f
    if (varianceEncodedInTarget) {
      // variance is encoded in target, we simply need to retore the offset
      // predictions.
      decodeCenterX = bbox.valueAt(i, 1) * priorWidth + pCenterX
      decodeCenterY = bbox.valueAt(i, 2) * priorHeight + pCenterY
      decodeWidth = Math.exp(bbox.valueAt(i, 3)).toFloat * priorWidth
      decodedHeight = Math.exp(bbox.valueAt(i, 4)).toFloat * priorHeight
    } else {
      // variance is encoded in bbox, we need to scale the offset accordingly.
      decodeCenterX = priorVariance.valueAt(i, 1) * bbox.valueAt(i, 1) * priorWidth + pCenterX
      decodeCenterY = priorVariance.valueAt(i, 2) * bbox.valueAt(i, 2) * priorHeight + pCenterY
      decodeWidth = Math.exp(priorVariance.valueAt(i, 3) * bbox.valueAt(i, 3)).toFloat * priorWidth
      decodedHeight = Math.exp(priorVariance.valueAt(i, 4) * bbox.valueAt(i, 4))
        .toFloat * priorHeight
    }
    decodedBoxes.setValue(i, 1, decodeCenterX - decodeWidth / 2)
    decodedBoxes.setValue(i, 2, decodeCenterY - decodedHeight / 2)
    decodedBoxes.setValue(i, 3, decodeCenterX + decodeWidth / 2)
    decodedBoxes.setValue(i, 4, decodeCenterY + decodedHeight / 2)
    if (isClipBoxes) {
      clipBoxes(decodedBoxes)
    }
  }

  def clipBoxes(bboxes: Tensor[Float]): Tensor[Float] = {
    bboxes.cmax(0).apply1(x => Math.min(1, x))
  }

  /**
   *
   * @param scoresNms N
   * @param bboxNms N * 4
   * @param scoresAll M
   * @param bboxAll M * 4
   * @return
   */
  def bboxVote(scoresNms: Tensor[Float], bboxNms: Tensor[Float],
    scoresAll: Tensor[Float], bboxAll: Tensor[Float],
    areasBuf: Tensor[Float] = null): RoiLabel = {
    var accBox: Tensor[Float] = null
    var accScore = 0f
    var box: Tensor[Float] = null
    val areasAll = if (areasBuf == null) {
      Tensor[Float]
    } else areasBuf
    getAreas(bboxAll, areasAll)
    var i = 1
    while (i <= scoresNms.size(1)) {
      box = bboxNms(i)
      if (accBox == null) {
        accBox = Tensor[Float](4)
      } else {
        accBox.fill(0f)
      }
      accScore = 0f
      var m = 1
      while (m <= scoresAll.size(1)) {
        val boxA = bboxAll(m)
        val iw = Math.min(box.valueAt(3), boxA.valueAt(3)) -
          Math.max(box.valueAt(1), boxA.valueAt(1)) + 1
        val ih = Math.min(box.valueAt(4), boxA.valueAt(4)) -
          Math.max(box.valueAt(2), boxA.valueAt(2)) + 1

        if (iw > 0 && ih > 0) {
          val ua = getArea(box) + areasAll.valueAt(m) - iw * ih
          val ov = iw * ih / ua
          if (ov >= 0.5) {
            accBox.add(scoresAll.valueAt(m), boxA)
            accScore += scoresAll.valueAt(m)
          }
        }
        m += 1
      }
      var x = 1
      while (x <= 4) {
        bboxNms.setValue(i, x, accBox.valueAt(x) / accScore)
        x += 1
      }
      i += 1
    }
    RoiLabel(scoresNms, bboxNms)
  }

  private def getArea(box: Tensor[Float]): Float = {
    require(box.dim() == 1 && box.nElement() >= 4)
    (box.valueAt(3) - box.valueAt(1) + 1) * (box.valueAt(4) - box.valueAt(2) + 1)
  }

  /**
   * get the areas of boxes
   * @param boxes N * 4 tensor
   * @param areas buffer to store the results
   * @return areas array
   */
  def getAreas(boxes: Tensor[Float], areas: Tensor[Float], startInd: Int = 1,
    normalized: Boolean = false): Tensor[Float] = {
    if (boxes.nElement() == 0) return areas
    require(boxes.size(2) >= 4)
    areas.resize(boxes.size(1))
    val boxesArr = boxes.storage().array()
    val offset = boxes.storageOffset() - 1
    val rowLength = boxes.stride(1)
    var i = 0
    var boffset = offset + startInd - 1
    while (i < boxes.size(1)) {
      val x1 = boxesArr(boffset)
      val y1 = boxesArr(boffset + 1)
      val x2 = boxesArr(boffset + 2)
      val y2 = boxesArr(boffset + 3)
      if (normalized) areas.setValue(i + 1, (x2 - x1) * (y2 - y1))
      else areas.setValue(i + 1, (x2 - x1 + 1) * (y2 - y1 + 1))
      boffset += rowLength
      i += 1
    }
    areas
  }

  def getGroundTruths(result: Tensor[Float]): Map[Int, Tensor[Float]] = {
    val indices = getGroundTruthIndices(result).toArray.sortBy(_._1)
    var gtMap = Map[Int, Tensor[Float]]()
    var ind = 0
    val iter = indices.iterator
    while (iter.hasNext) {
      val x = iter.next()
      val gt = result.narrow(1, x._2._1, x._2._2)
      // -1 represent those images without label
      if (gt.size(1) > 1 || gt.valueAt(1, 2) != -1) {
        gtMap += (ind -> gt)
      }
      ind += 1
    }
    gtMap
    //    indices.map(x => x._1 -> result.narrow(1, x._2._1, x._2._2))
  }

  def getGroundTruthIndices(result: Tensor[Float]): Map[Int, (Int, Int)] = {
    var indices = Map[Int, (Int, Int)]()
    if (result.nElement() == 0) return indices
    var prev = -1f
    var i = 1
    var start = 1
    if (result.size(1) == 1) {
      indices += (result.valueAt(i, 1).toInt -> (1, 1))
      return indices
    }
    while (i <= result.size(1)) {
      if (prev != result.valueAt(i, 1)) {
        if (prev >= 0) {
          indices += (prev.toInt -> (start, i - start))
        }
        start = i
      }
      prev = result.valueAt(i, 1)
      if (i == result.size(1)) {
        indices += (prev.toInt -> (start, i - start + 1))
      }
      i += 1
    }
    indices
  }

  private def decodeSignalBoxWithWeight(encodeBox: Tensor[Float], bbox: Tensor[Float],
             weight: Array[Float], decodeBox: Tensor[Float]): Unit = {
    require(bbox.nDimension() == 1 && encodeBox.nDimension() == 1 && decodeBox.dim() == 1,
    s"Only support decode single bbox, but " +
      s"get ${bbox.nDimension()}, ${encodeBox.nDimension()}, ${decodeBox.dim()}")

    require(encodeBox.nElement() == decodeBox.nElement(), s"element number of encode tensor" +
      s" and decode tensor should be same, but get ${encodeBox.nElement()} ${decodeBox.nElement()}")

    val TO_REMOVE = 1 // refer to pytorch, maybe it will be removed in future
    val x1 = bbox.valueAt(1)
    val y1 = bbox.valueAt(2)
    val x2 = bbox.valueAt(3)
    val y2 = bbox.valueAt(4)
    val priorWidth = x2 - x1 + TO_REMOVE
    val priorHight = y2 - y1 + TO_REMOVE
    val pCenterX = x1 + priorWidth/ 2
    val pCenterY = y1 + priorHight / 2

    val wx = weight(0)
    val wy = weight(1)
    val ww = weight(2)
    val wh = weight(3)

    encodeBox.resize(Array(encodeBox.nElement() / 4, 4))

    // copy for contigious
    val dx = encodeBox.select(2, 1).contiguous().div(wx)
    val dy = encodeBox.select(2, 2).contiguous().div(wy)
    val dw = encodeBox.select(2, 3).contiguous().div(ww)
    val dh = encodeBox.select(2, 4).contiguous().div(wh)

    // not change original input
    encodeBox.resize(encodeBox.nElement())

    // clamp, dw,  dh
    val bboxClip = 62.5f
    clamp(dw, 0.0f, bboxClip)
    clamp(dh, 0.0f, bboxClip)

    val pred_ctr_x = dx * priorWidth + pCenterX
    val pred_ctr_y = dy * priorHight + pCenterY

    val pred_w = dw.exp().mul(priorWidth).mul(0.5f)
    val pred_h = dh.exp().mul(priorHight).mul(0.5f)

    // todo: memory optimzation
    val buffer1 = Tensor[Float]().resizeAs(pred_ctr_x).copy(pred_ctr_x).sub(pred_w)
    val buffer2 = Tensor[Float]().resizeAs(pred_ctr_y).copy(pred_ctr_y).sub(pred_h)
    val buffer3 = Tensor[Float]().resizeAs(pred_ctr_x).copy(pred_ctr_x).add(pred_w).add(-1.0f)
    val buffer4 = Tensor[Float]().resizeAs(pred_ctr_y).copy(pred_ctr_y).add(pred_h).add(-1.0f)

    val arrBuffer1 = buffer1.storage().array()
    val arrBuffer2 = buffer2.storage().array()
    val arrBuffer3 = buffer3.storage().array()
    val arrBuffer4 = buffer4.storage().array()
    val arrBox = decodeBox.storage().array()
    val offset = decodeBox.storageOffset() - 1

    var i = 0
    var j = 0
    while (i < arrBuffer1.length) {
      arrBox(j + offset) = arrBuffer1(i)
      arrBox(j + 1 + offset) = arrBuffer2(i)
      arrBox(j + 2 + offset) = arrBuffer3(i)
      arrBox(j + 3 + offset) = arrBuffer4(i)
      i += 1
      j += 4
    }
  }

  def decodeWithWeight(encodeBox: Tensor[Float], bbox: Tensor[Float],
             weight: Array[Float], decodeBox: Tensor[Float]): Unit = {
    require(encodeBox.size(1) == bbox.size(1))
    require(encodeBox.size(1) == decodeBox.size(1))

    val numBboxes = bbox.size(1)
    if (numBboxes > 0) require(bbox.size(2) == 4)

    var i = 1
    while (i <= numBboxes) {
      decodeSignalBoxWithWeight(encodeBox.select(1, i), bbox.select(1, i),
        weight, decodeBox.select(1, i))
      i += 1
    }
  }

  private def clamp(input: Tensor[Float], min: Float, max: Float): Unit = {
    require(input.isContiguous(), "input for clamp should be contiguous")
    val arr = input.storage().array()
    val offset = input.storageOffset() - 1
    var i = 0
    while (i < arr.length) {
      val value = arr(i)
      if (value < min) arr(i) = value
      if (value > max) arr(i) = max
      i += 1
    }
  }
}
