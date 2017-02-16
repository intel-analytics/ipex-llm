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

package com.intel.analytics.bigdl.models.fasterrcnn.utils

import com.intel.analytics.bigdl.tensor.Tensor

/**
 * Non-Maximum Suppression (nms) for Object Detection
 * The goal of nms is to solve the problem that groups of several detections near the real location,
 * ideally obtaining only one detection per object
 */
class Nms extends Serializable {

  @transient private var areas: Tensor[Float] = _
  @transient private var sortedScores: Tensor[Float] = _
  @transient private var sortedInds: Tensor[Float] = _
  @transient private var sortIndBuffer: Array[Int] = _
  @transient private var suppressed: Array[Int] = _

  private def init(size: Int): Unit = {
    if (suppressed == null || suppressed.length < size) {
      suppressed = new Array[Int](size)
      sortIndBuffer = new Array[Int](size)
    } else {
      var i = 0
      while (i < size) {
        suppressed(i) = 0
        i += 1
      }
    }
    if (sortedScores == null) {
      sortedScores = Tensor[Float]
      sortedInds = Tensor[Float]
      areas = Tensor[Float]
    }
  }

  /**
   * 1. first sort the scores from highest to lowest and get indices
   * 2. for the bbox of first index,
   * get the overlap between this box and the remaining bboxes
   * put the first index to result buffer
   * 3. update the indices by keeping those bboxes with overlap less than thresh
   * 4. repeat 2 and 3 until the indices are empty
   * @param x1      x1 tensor
   * @param y1      y1 tensor
   * @param x2      x2 tensor
   * @param y2      y2 tensor
   * @param scores  score tensor
   * @param thresh  overlap thresh
   * @param indices buffer to store indices after nms
   * @return the length of indices after nms
   */
  private def nms(x1: Tensor[Float], y1: Tensor[Float], x2: Tensor[Float], y2: Tensor[Float],
    scores: Tensor[Float], thresh: Float, indices: Array[Int]): Int = {
    if (scores.nElement() == 0) return 0
    require(indices.length >= scores.nElement())
    require(x1.isContiguous())

    init(scores.nElement())
    getAreas(x1, x2, y1, y2, areas)
    // indices start from 1
    val orderLength = getSortedScoreInds(scores, sortIndBuffer)
    var indexLenth = 0
    var i = 0
    var curInd = 0

    while (i < orderLength) {
      def nmsOne(): Unit = {
        curInd = sortIndBuffer(i)
        if (suppressed(curInd - 1) == 1) {
          return
        }
        indices(indexLenth) = curInd
        indexLenth += 1
        var k = i + 1
        while (k < orderLength) {
          if (suppressed(sortIndBuffer(k) - 1) != 1 &&
            isOverlapRatioGtThresh(x1, x2, y1, y2, curInd, sortIndBuffer(k), thresh)) {
            suppressed(sortIndBuffer(k) - 1) = 1
          }
          k += 1
        }
      }
      nmsOne()
      i += 1
    }
    indexLenth
  }

  private def getSortedScoreInds(scores: Tensor[Float], resultBuffer: Array[Int]): Int = {
    // note that when the score is the same,
    // the order of the indices are different in python and here
    scores.topk(scores.nElement(), dim = 1, increase = false, result = sortedScores,
      indices = sortedInds
    )
    var i = 0
    while (i < scores.nElement()) {
      sortIndBuffer(i) = sortedInds.valueAt(i + 1).toInt
      i += 1
    }
    scores.nElement()
  }

  @transient var buffer: Tensor[Float] = _

  private def getAreas(x1: Tensor[Float], x2: Tensor[Float],
    y1: Tensor[Float], y2: Tensor[Float], areas: Tensor[Float]): Tensor[Float] = {
    if (buffer == null) buffer = Tensor[Float]
    // (x2 - x1 + 1) * (y2 - y1 + 1)
    areas.resizeAs(x2).add(x2, -1, x1).add(1)
    buffer.resizeAs(y2).add(y2, -1, y1).add(1)
    areas.cmul(buffer)
    areas
  }

  private def isOverlapRatioGtThresh(x1: Tensor[Float], x2: Tensor[Float],
    y1: Tensor[Float], y2: Tensor[Float], ind: Int, ind2: Int, thresh: Float): Boolean = {
    if (x1.valueAt(ind2) >= x2.valueAt(ind) ||
      x1.valueAt(ind) >= x2.valueAt(ind2) ||
      y1.valueAt(ind2) >= y2.valueAt(ind) ||
      y1.valueAt(ind) >= y2.valueAt(ind2)) {
      return false
    }
    val w = math.min(x2.valueAt(ind2), x2.valueAt(ind)) -
      math.max(x1.valueAt(ind2), x1.valueAt(ind)) + 1
    val h = math.min(y2.valueAt(ind2), y2.valueAt(ind)) -
      math.max(y1.valueAt(ind2), y1.valueAt(ind)) + 1
    val overlap = w * h
    overlap / ((areas.valueAt(ind2) + areas.valueAt(ind)) - overlap) > thresh
  }


  /**
   *
   * @param dets N*5
   * @param thresh
   * @return
   */
  def nms(dets: Tensor[Float], thresh: Float): Array[Int] = {
    if (dets.nElement() == 0) return Array[Int]()
    val indexes = new Array[Int](dets.size(1))
    val keepNum = nms(dets.select(2, 5), dets.narrow(2, 1, 4).t(), thresh,
      indexes)
    indexes.slice(0, keepNum)
  }

  @transient var x1Buffer: Tensor[Float] = _
  @transient var x2Buffer: Tensor[Float] = _
  @transient var y1Buffer: Tensor[Float] = _
  @transient var y2Buffer: Tensor[Float] = _

  /**
   *
   * @param scores score tensor, size N
   * @param bbox   bboxes 4*N
   * @param thresh overlap thresh
   * @return the length of indices after nms
   */
  def nms(scores: Tensor[Float], bbox: Tensor[Float], thresh: Float,
    indsBuffer: Array[Int]): Int = {
    if (scores.nElement() == 0) return 0
    val (x1, y1, x2, y2) = if (!bbox.isContiguous()) {
      if (x1Buffer == null) {
        x1Buffer = Tensor[Float]
        x2Buffer = Tensor[Float]
        y1Buffer = Tensor[Float]
        y2Buffer = Tensor[Float]
      }
      x1Buffer.resize(bbox.size(2)).copy(bbox(1))
      x2Buffer.resize(bbox.size(2)).copy(bbox(2))
      y1Buffer.resize(bbox.size(2)).copy(bbox(3))
      y2Buffer.resize(bbox.size(2)).copy(bbox(4))
      (x1Buffer, x2Buffer, y1Buffer, y2Buffer)
    } else {
      (bbox(1), bbox(2), bbox(3), bbox(4))
    }
    nms(x1, y1, x2, y2, scores, thresh, indsBuffer)
  }
}
