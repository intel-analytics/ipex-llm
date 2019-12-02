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

import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import scala.collection.mutable.ArrayBuffer

/**
 * Generates a regular grid of multi-scale, multi-aspect anchor boxes.
 */
class Anchor(ratios: Array[Float], scales: Array[Float]) extends Serializable {

  private var baseSize = 16
  private var basicAnchors: Tensor[Float] = generateBasicAnchors(ratios, scales, baseSize)

  val anchorNum = ratios.length * scales.length
  /**
   * first generate shiftX and shiftY over the whole feature map
   * then apply shifts for each basic anchors
   * @param width      feature map width
   * @param height     feature map height
   * @param featStride stride to move
   * @return all anchors over the feature map
   */
  def generateAnchors(width: Int, height: Int, featStride: Float = 16): Tensor[Float] = {
    val (shiftX, shiftY) = generateShifts(width, height, featStride)
    if (featStride != baseSize) {
      basicAnchors = generateBasicAnchors(ratios, scales, featStride)
      baseSize = featStride.toInt
    }
    getAllAnchors(shiftX, shiftY, basicAnchors)
  }

  /**
   * Here, we generate anchors without change area.
   * @param ratios
   * @param scales
   * @param baseSize stride to move
   * @return anchors with shape (ratios number * scales number, 4).
   *         And element order is (-width / 2, -height / 2, width / 2, height / 2)
   */
  private def generateBasicAnchors(ratios: Array[Float], scales: Array[Float],
    baseSize: Float = 16): Tensor[Float] = {
    val anchors = new ArrayBuffer[Float]
    for (i <- 0 until scales.length) {
      val area = math.pow(scales(i) * baseSize, 2)
      for (j <- 0 until ratios.length) {
        val w = math.sqrt(area / ratios(j)).toFloat
        val h = ratios(j) * w.toFloat
        val halfW = w / 2.0f
        val halfH = h / 2.0f

        anchors.append(-halfW)
        anchors.append(-halfH)
        anchors.append(halfW)
        anchors.append(halfH)
      }
    }
    Tensor[Float](data = anchors.toArray, shape = Array[Int](ratios.length * scales.length, 4))
  }

  @transient private var shiftX: Tensor[Float] = _
  @transient private var shiftY: Tensor[Float] = _

  /**
   * generate shifts wrt width, height and featStride
   * in order to generate anchors over the whole feature map
   * @param width      feature map width
   * @param height     feature map height
   * @param featStride stride to move
   * @return shiftX and shiftY
   */
  private[nn] def generateShifts(width: Int, height: Int, featStride: Float):
  (Tensor[Float], Tensor[Float]) = {
    if (shiftX == null) {
      shiftX = Tensor[Float]
      shiftY = Tensor[Float]
    }
    var i = -1
    shiftX.resize(width).apply1 { x => i += 1; i * featStride } // 0, f, 2f, ..., wf
    i = -1
    shiftY.resize(height).apply1 { x => i += 1; i * featStride } // 0, f, 2f, ..., hf
    (shiftX, shiftY)
  }

  @transient private var allAnchors: Tensor[Float] = _

  /**
   * each anchor add with shiftX and shiftY
   * @param shiftX  a list of shift in X direction
   * @param shiftY  a list of shift in Y direction
   * @param anchors basic anchors that will apply shifts
   * @return anchors with all shifts
   */
  private def getAllAnchors(shiftX: Tensor[Float], shiftY: Tensor[Float],
    anchors: Tensor[Float]): Tensor[Float] = {
    if (allAnchors == null) {
      allAnchors = Tensor[Float]
    }
    val S = shiftX.nElement() * shiftY.nElement()
    val A = anchors.size(1)
    allAnchors.resize(S * A, 4)
    val xsArr = shiftX.storage().array()
    val ysArr = shiftY.storage().array()
    val allAnchorArr = allAnchors.storage().array()
    var aOffset = allAnchors.storageOffset() - 1
    val anchorArr = anchors.storage().array()
    var ysOffset = shiftY.storageOffset() - 1
    var ys = 0
    while (ys < shiftY.nElement()) {
      var xs = 0
      var xsOffset = shiftX.storageOffset() - 1
      while (xs < shiftX.nElement()) {
        var a = 0
        var anchorOffset = anchors.storageOffset() - 1
        while (a < A) {
          allAnchorArr(aOffset) = anchorArr(anchorOffset) + xsArr(xsOffset)
          allAnchorArr(aOffset + 1) = anchorArr(anchorOffset + 1) + ysArr(ysOffset)
          allAnchorArr(aOffset + 2) = anchorArr(anchorOffset + 2) + xsArr(xsOffset)
          allAnchorArr(aOffset + 3) = anchorArr(anchorOffset + 3) + ysArr(ysOffset)
          aOffset += 4
          anchorOffset += 4
          a += 1
        }
        xs += 1
        xsOffset += 1
      }
      ys += 1
      ysOffset += 1
    }
    allAnchors
  }

  /**
   * Given a vector of widths (ws) and heights (hs) around a center
   * (x_ctr, y_ctr), output a set of anchors (windows).
   * note that the value of ws and hs is changed after mkAnchors (half)
   * x1 = xCtr - (ws-1)/2 = xCtr - ws/2 + 0.5
   * y1 = yCtr - (hs-1)/2 = yCtr - hs/2 + 0.5
   * x2 = xCtr + (ws-1)/2 = xCtr + ws/2 - 0.5
   * y2 = yCtr + (hs-1)/2 = yCtr + hs/2 - 0.5
   * @param ws   widths
   * @param hs   heights
   * @param xCtr center x
   * @param yCtr center y
   * @return anchors around this center, with shape (4, N)
   */
  private def mkAnchors(ws: Tensor[Float], hs: Tensor[Float],
    xCtr: Float, yCtr: Float): Tensor[Float] = {
    require(ws.size(1) == hs.size(1))
    val anchors = Tensor(ws.size(1), 4)
    var i = 1
    while (i <= ws.size(1)) {
      val w = ws.valueAt(i) / 2 - 0.5f
      val h = hs.valueAt(i) / 2 - 0.5f
      anchors.setValue(i, 1, xCtr - w)
      anchors.setValue(i, 2, yCtr - h)
      anchors.setValue(i, 3, xCtr + w)
      anchors.setValue(i, 4, yCtr + h)
      i += 1
    }
    anchors
  }

  /**
   * Return width, height, x center, and y center for an anchor (window).
   */
  private def getBasicAchorInfo(anchor: Tensor[Float]): (Float, Float, Float, Float) = {
    val w = anchor.valueAt(3) - anchor.valueAt(1) + 1
    val h = anchor.valueAt(4) - anchor.valueAt(2) + 1
    val xCtr = anchor.valueAt(1) + 0.5f * (w - 1)
    val yCtr = anchor.valueAt(2) + 0.5f * (h - 1)
    (w, h, xCtr, yCtr)
  }

  @transient var ws: Tensor[Float] = _
  @transient var hs: Tensor[Float] = _

  /**
   * Enumerate a set of anchors for each aspect ratio with respect to an anchor.
   * ratio = height / width
   */
  private def ratioEnum(anchor: Tensor[Float], ratios: Tensor[Float]): Tensor[Float] = {
    val (width, height, xCtr, yCtr) = getBasicAchorInfo(anchor)
    val area = width * height
    if (ws == null) {
      ws = Tensor()
      hs = Tensor()
    }
    // get a set of widths
    ws.resizeAs(ratios).map(ratios, (w, ratio) => Math.sqrt(area / ratio).round)
    // get corresponding heights
    hs.resizeAs(ws).cmul(ws, ratios).apply1(Math.round)
    mkAnchors(ws, hs, xCtr, yCtr)
  }

  /**
   * Enumerate a set of anchors for each scale wrt an anchor.
   */
  private def scaleEnum(anchor: Tensor[Float], scales: Tensor[Float]): Tensor[Float] = {
    if (ws == null) {
      ws = Tensor()
      hs = Tensor()
    }
    val (width, height, xCtr, yCtr) = getBasicAchorInfo(anchor)
    ws.resizeAs(scales).mul(scales, width)
    hs.resizeAs(scales).mul(scales, height)
    mkAnchors(ws, hs, xCtr, yCtr)
  }
}

object Anchor {
  def apply(ratios: Array[Float], scales: Array[Float]): Anchor = new Anchor(ratios, scales)
}





