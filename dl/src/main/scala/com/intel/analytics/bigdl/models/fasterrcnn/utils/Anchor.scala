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

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

case class AnchorParam(ratios: Array[Float], scales: Array[Float]) {
  val num: Int = ratios.length * scales.length
}

class Anchor(param: AnchorParam) extends Serializable {

  val basicAnchors: Tensor[Float] = generateBasicAnchors(param.ratios, param.scales)

  /**
   * Generate anchor (reference) windows by enumerating aspect ratios X
   * scales wrt a reference (0, 0, 15, 15) window.
   *
   */
  def generateBasicAnchors(ratios: Array[Float], scales: Array[Float],
    baseSize: Float = 16): Tensor[Float] = {
    val baseAnchor = Tensor(Storage(Array(0, 0, baseSize - 1, baseSize - 1)))
    val ratioAnchors = ratioEnum(baseAnchor, Tensor(Storage(ratios)))
    val anchors = Tensor[Float](4, scales.length * ratioAnchors.size(1))
    var idx = 1
    var i = 1
    while (i <= ratioAnchors.size(1)) {
      val scaleAnchors = scaleEnum(ratioAnchors(i), Tensor(Storage(scales)))
      var j = 1
      while (j <= scaleAnchors.size(1)) {
        anchors.setValue(1, idx, scaleAnchors.valueAt(j, 1))
        anchors.setValue(2, idx, scaleAnchors.valueAt(j, 2))
        anchors.setValue(3, idx, scaleAnchors.valueAt(j, 3))
        anchors.setValue(4, idx, scaleAnchors.valueAt(j, 4))
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
   *
   */
  private def mkanchors(ws: Tensor[Float], hs: Tensor[Float],
    xCtr: Float, yCtr: Float): Tensor[Float] = {
    val a1 = ws.-(1).mul(-0.5f).add(xCtr)
    val a2 = hs.-(1).mul(-0.5f).add(yCtr)
    val a3 = ws.-(1).mul(0.5f).add(xCtr)
    val a4 = hs.-(1).mul(0.5f).add(yCtr)
    val anchors = Tensor[Float](a1.nElement(), 4)
    var i = 1
    while (i <= a1.nElement()) {
      anchors.setValue(i, 1, a1.valueAt(i))
      anchors.setValue(i, 2, a2.valueAt(i))
      anchors.setValue(i, 3, a3.valueAt(i))
      anchors.setValue(i, 4, a4.valueAt(i))
      i += 1
    }
    anchors
  }

  /**
   * Return width, height, x center, and y center for an anchor (window).
   */
  private def whctrs(anchor: Tensor[Float]): Array[Float] = {
    val w = anchor.valueAt(3) - anchor.valueAt(1) + 1
    val h = anchor.valueAt(4) - anchor.valueAt(2) + 1
    val xCtr = anchor.valueAt(1) + 0.5f * (w - 1)
    val yCtr = anchor.valueAt(2) + 0.5f * (h - 1)
    Array[Float](w, h, xCtr, yCtr)
  }

  @transient var ws: Tensor[Float] = null
  @transient var hs: Tensor[Float] = null

  /**
   * Enumerate a set of anchors for each aspect ratio wrt an anchor.
   *
   */
  private def ratioEnum(anchor: Tensor[Float], ratios: Tensor[Float]): Tensor[Float] = {
    // w, h, x_ctr, y_ctr
    val out = whctrs(anchor)
    val size = out(0) * out(1)
    if (ws == null) {
      ws = Tensor[Float]
      hs = Tensor[Float]
    }
    ws.resizeAs(ratios).copy(ratios).apply1(x => Math.sqrt(size / x).round)
    hs.resizeAs(ws).copy(ws).map(ratios, (w, r) => Math.round(w * r))
    mkanchors(ws, hs, out(2), out(3))
  }

  /**
   * Enumerate a set of anchors for each scale wrt an anchor.
   *
   */
  private def scaleEnum(anchor: Tensor[Float], scales: Tensor[Float]): Tensor[Float] = {
    if (ws == null) {
      ws = Tensor[Float]
      hs = Tensor[Float]
    }
    val out = whctrs(anchor)
    ws.resizeAs(scales).copy(scales).apply1(x => x * out(0))
    hs.resizeAs(scales).copy(scales).apply1(x => x * out(1))
    mkanchors(ws, hs, out(2), out(3))
  }


  @transient var shifts: Tensor[Float] = null
  @transient var shiftX: Tensor[Float] = null
  @transient var shiftY: Tensor[Float] = null

  def generateShifts(width: Int, height: Int, featStride: Float): Tensor[Float] = {
    if (shiftX == null) {
      shiftX = Tensor[Float]
      shiftY = Tensor[Float]
    }
    shiftX.resize(width)
    shiftY.resize(height)
    var i = 0
    while (i < width) {
      shiftX.setValue(i + 1, i * featStride)
      i += 1
    }
    i = 0
    while (i < height) {
      shiftY.setValue(i + 1, i * featStride)
      i += 1
    }
    if (shifts == null) {
      shifts = Tensor[Float]
    }
    i = 1
    val total = shiftX.size(1) * shiftY.size(1)
    shifts.resize(2, total)
    var s = 1
    var y = 0f
    shifts(1).copy(shiftX.view(1, shiftX.size(1)).expand(Array(shiftY.size(1), shiftX.size(1))))
    shifts(2).copy(shiftY.view(shiftY.size(1), 1).expand(Array(shiftY.size(1), shiftX.size(1))))
    shifts
  }


  @transient var allAnchors: Tensor[Float] = _

  /**
   * each row of shifts add each row of anchors
   * and return shifts.size(1) * anchors.size(1) rows
   * @param shifts  2 * N
   * @param anchors 4 * M
   * @return
   */
  def getAllAnchors(shifts: Tensor[Float],
    anchors: Tensor[Float]): Tensor[Float] = {
    require(shifts.size(1) == 2 && anchors.size(1) == 4)
    if (allAnchors == null) {
      allAnchors = Tensor[Float]
    }
    val totalN = shifts.size(2) * anchors.size(2)
    allAnchors.resize(4, totalN)
    var i = 1
    while ( i <= 4) {
      val s = if (i == 1 || i == 3) 1 else 2
      val expandShifts =
        shifts(s).view(shifts.size(2), 1).expand(Array(shifts.size(2), anchors.size(2)))
          .reshape(Array(totalN))
      val expandAnchors = anchors(i).view(1, anchors.size(2))
        .expand(Array(shifts.size(2), anchors.size(2))).reshape(Array(totalN))
      allAnchors(i).add(expandShifts, expandAnchors)
      i += 1
    }
    allAnchors
  }

  def getAllAnchors(width: Int, height: Int, featStride: Float): Tensor[Float] = {
    val shifts = generateShifts(width, height, featStride)
    getAllAnchors(shifts, basicAnchors)
  }

}
