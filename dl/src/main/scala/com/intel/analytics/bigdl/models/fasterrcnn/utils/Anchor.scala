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

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

/**
 * Generates a regular grid of multi-scale, multi-aspect anchor boxes.
 * @param param (anchor ratios, anchor scales)
 */
class Anchor(param: AnchorParam)(implicit ev: TensorNumeric[Float]) extends Serializable {

  val basicAnchors: Tensor[Float] = generateBasicAnchors(param.ratios, param.scales)

  /**
   * first generate shiftX and shiftY over the whole feature map
   * then apply shifts for each basic anchors
   * @param width      feature map width
   * @param height     feature map height
   * @param featStride stride to move
   * @return all anchors over the feature map
   */
  def generateAnchors(width: Int, height: Int, featStride: Float): Tensor[Float] = {
    val (shiftX, shiftY) = generateShifts(width, height, featStride)
    getAllAnchors(shiftX, shiftY, basicAnchors)
  }

  @transient var shiftX: Tensor[Float] = _
  @transient var shiftY: Tensor[Float] = _

  /**
   * generate shifts wrt width, height and featStride
   * in order to generate anchors over the whole feature map
   * @param width      feature map width
   * @param height     feature map height
   * @param featStride stride to move
   * @return shiftX and shiftY
   */
  def generateShifts(width: Int, height: Int, featStride: Float):
  (Tensor[Float], Tensor[Float]) = {
    if (shiftX == null) {
      shiftX = Tensor()
      shiftY = Tensor()
    }
    var i = -1
    shiftX.resize(1, width).apply1 { x => i += 1; i * featStride } // 0, f, 2f, ..., wf
    i = -1
    shiftY.resize(height, 1).apply1 { x => i += 1; i * featStride } // 0, f, 2f, ..., hf
    // 0, f, 2f, ..., wf, 0, f, 2f, ..., wf, 0, f, 2f, ..., wf, ......
    // each 0, f, 2f, ..., wf repeat height times
    val expandedX = shiftX.expand(Array(height, width)).reshape(Array(height * width, 1))
    // 0, 0, 0, ..., 0, f, f, f, ..., f, 2f, 2f, 2f, ..., 2f, ......,  hf, hf, hf, ..., hf
    // each xf repeat width times
    val expandedY = shiftY.expand(Array(height, width)).reshape(Array(height * width, 1))
    (expandedX, expandedY)
  }

  @transient var allAnchors: Tensor[Float] = _

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
    val S = shiftX.size(1)
    val A = anchors.size(2)
    allAnchors.resize(4, S * A)
    val outputData = allAnchors.storage().array()
    var aOffset = allAnchors.storageOffset() - 1
    val anchorData = anchors.storage().array()
    var i = 1
    while (i <= 4) {
      // 1 for shiftX and 2 for shiftY
      val shift = if (i == 1 || i == 3) shiftX else shiftY
      val expandShifts = shift.view(S, 1).expand(Array(S, A))
      allAnchors(i).copy(expandShifts)
      var s = 0
      while (s < S) {
        ev.vAdd(A, outputData, aOffset, anchorData, (i - 1) * A, outputData, aOffset)
        s += 1
        aOffset += A
      }
      i += 1
    }
    allAnchors
  }

  /**
   * Generate anchor (reference) windows by enumerating aspect ratios(M) X scales(N)
   * wrt a reference (0, 0, 15, 15) window.
   * 1. generate anchors for different ratios (4, M)
   * 2. for each anchors generated in 1, scale them to get scaled anchors (4, M*N)
   */
  def generateBasicAnchors(ratios: Tensor[Float], scales: Tensor[Float],
    baseSize: Float = 16): Tensor[Float] = {
    val baseAnchor = Tensor(Storage(Array(0, 0, baseSize - 1, baseSize - 1)))
    val ratioAnchors = ratioEnum(baseAnchor, ratios)
    val anchors = Tensor(4, scales.size(1) * ratioAnchors.size(2))
    var idx = 1
    var i = 1
    while (i <= ratioAnchors.size(2)) {
      val scaleAnchors = scaleEnum(ratioAnchors.select(2, i), scales)
      var j = 1
      while (j <= scaleAnchors.size(2)) {
        anchors.setValue(1, idx, scaleAnchors.valueAt(1, j))
        anchors.setValue(2, idx, scaleAnchors.valueAt(2, j))
        anchors.setValue(3, idx, scaleAnchors.valueAt(3, j))
        anchors.setValue(4, idx, scaleAnchors.valueAt(4, j))
        idx = idx + 1
        j += 1
      }
      i += 1
    }
    anchors
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
    val anchors = Tensor(4, ws.size(1))
    ws.mul(0.5f)
    hs.mul(0.5f)
    anchors(1).fill(xCtr).add(-1, ws).add(0.5f)
    anchors(2).fill(yCtr).add(-1, hs).add(0.5f)
    anchors(3).fill(xCtr).add(ws).add(-0.5f)
    anchors(4).fill(yCtr).add(hs).add(-0.5f)
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
    ws.resizeAs(ratios).copy(ratios).apply1(ratio => Math.sqrt(area / ratio).round)
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

/**
 * anchor parameters
 * @param _ratios ratio = height / width
 * @param _scales scale of width and height
 */
case class AnchorParam(_ratios: Array[Float], _scales: Array[Float]) {
  val num: Int = _ratios.length * _scales.length
  val ratios: Tensor[Float] = Tensor(Storage(_ratios))
  val scales: Tensor[Float] = Tensor(Storage(_scales))
}

