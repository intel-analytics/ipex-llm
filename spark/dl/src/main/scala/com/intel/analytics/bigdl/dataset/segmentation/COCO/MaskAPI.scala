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

package com.intel.analytics.bigdl.dataset.segmentation.COCO

import com.intel.analytics.bigdl.transform.vision.image.label.roi.{PolyMasks, RLEMasks}
import scala.collection.mutable.ArrayBuffer

object MaskAPI {

  /**
   * Convert an unsigned int to long (note: int may overflow)
   *
   * @param i
   * @return
   */
  def uint2long(i: Int): Long = {
    if (i >= 0) {
      i
    } else {
      i.toLong - Int.MinValue.toLong + Int.MaxValue.toLong + 1
    }
  }

  // scalastyle:off methodName
  def RLE2String(rle: RLEMasks): String = {
    // Similar to LEB128 but using 6 bits/char and ascii chars 48-111.
    val m = rle.counts.length
    val s = new ArrayBuffer[Char]()
    for (i <- 0 until m) {
      var x = rle.get(i)
      if (i > 2) x -= rle.get(i - 2)
      var more = true
      while (more) {
        var c = (x & 0x1f)
        x >>= 5
        more = if ((c & 0x10) != 0) x != -1 else x != 0
        if (more) c |= 0x20
        c += 48
        s += c.toChar
      }
    }
    new String(s.toArray)
  }
  // scalastyle:on methodName


  def string2RLE(s: String, h: Int, w: Int): RLEMasks = {
    val cnts = new ArrayBuffer[Int]()
    var m = 0
    var p = 0
    while (p < s.length) {
      var x = 0L
      var k = 0
      var more = true
      while (more) {
        val c = s(p).toLong - 48
        x |= (c & 0x1f) << (5 * k)
        more = (c & 0x20) != 0
        k += 1
        p += 1
        if (!more && (c & 0x10) != 0) x |= -1 << (5 * k)
      }
      if (m > 2) x += uint2long(cnts(m - 2))
      cnts += x.toInt
      m += 1
    }
    RLEMasks(cnts.toArray, h, w)
  }

  def poly2RLE(poly: PolyMasks, height: Int, width: Int): Array[RLEMasks] = {
    poly.poly.map(xy => {
      // upsample and get discrete points densely along entire boundary
      val scale = 5d
      val (u, v, upsamplePoints) = {
        val nPoints = xy.length / 2
        val x = new Array[Long](nPoints + 1)
        val y = new Array[Long](nPoints + 1)
        for (j <- 0 until nPoints) {
          x(j) = Math.floor(scale * xy(j * 2 + 0) + .5).toLong
          y(j) = Math.floor(scale * xy(j * 2 + 1) + .5).toLong
        }
        x(nPoints) = x(0)
        y(nPoints) = y(0)
        val m1 = (0 until nPoints).map { case j =>
          Math.max(Math.abs(x(j) - x(j + 1)), Math.abs(y(j) - y(j + 1))) + 1
        }.sum.toInt
        val u = new Array[Long](m1)
        val v = new Array[Long](m1)

        var m = 0
        for (j <- 0 until nPoints) {
          val (xs, xe, ys, ye, dx, dy, flip) = {
            val _xs = x(j)
            val _xe = x(j + 1)
            val _ys = y(j)
            val _ye = y(j + 1)
            val _dx = Math.abs(_xe - _xs)
            val _dy = Math.abs(_ys - _ye)
            val _flip = (_dx >= _dy && _xs > _xe) || (_dx < _dy && _ys > _ye)
            if (_flip) (_xe, _xs, _ye, _ys, _dx, _dy, _flip)
            else (_xs, _xe, _ys, _ye, _dx, _dy, _flip)
          }

          if (dx >= dy) {
            for (d <- 0 to dx.toInt) {
              val s = (ye - ys).toDouble / dx
              val t = if (flip) dx - d else d
              u(m) = t + xs
              v(m) = Math.floor(ys + s * t + .5).toLong
              m += 1
            }
          }
          else {
            for (d <- 0 to dy.toInt) {
              val s = (xe - xs).toDouble / dy
              val t = if (flip) dy - d else d
              v(m) = t + ys
              u(m) = Math.floor(xs + s * t + .5).toLong
              m += 1
            }
          }
        }
        (u, v, m)
      }
      // get points along y-boundary and downsample
      val (downsampleX, downsampleY, downsamplePoints) = {
        // use an independent scope
        val nPoints = upsamplePoints
        var m = 0
        val x = new Array[Long](nPoints)
        val y = new Array[Long](nPoints)
        for (j <- 1 until nPoints) {
          if (u(j) != u(j - 1)) {
            // Should u(j) - 1 be u(j - 1) ????
            val _xd = if (u(j) < u(j - 1)) u(j) else u(j) - 1
            val xd = (_xd.toDouble + .5) / scale - .5
            if (Math.floor(xd) != xd || xd < 0 || xd > width - 1) {
              // continue
            } else {
              var yd = (if (v(j) < v(j - 1)) v(j) else v(j - 1)).toDouble
              yd = (yd + .5) / scale - .5
              if (yd < 0) {
                yd = 0
              } else if (yd > height) {
                yd = height
              }
              yd = Math.ceil(yd)
              x(m) = xd.toInt
              y(m) = yd.toInt
              m += 1
            }
          }
        }
        (x, y, m)
      }

      {
        // compute rle encoding given y-boundary points
        val x = downsampleX
        val y = downsampleY
        val nPoints = downsamplePoints + 1
        val a = new Array[Long](nPoints)
        for (j <- 0 until nPoints - 1)
          a(j) = x(j) * height + y(j)
        a(nPoints - 1) = height * width
        scala.util.Sorting.quickSort(a)

        var p = 0L
        for (j <- 0 until nPoints) {
          val t = a(j)
          a(j) -= p
          p = t
        }
        val b = new ArrayBuffer[Int]()
        var j = 1
        var m = 1
        b += a(0).toInt
        while (j < nPoints) {
          if (a(j) > 0) {
            b += a(j).toInt
            m += 1
            j += 1
          }
          else {
            j += 1
            if (j < nPoints) {
              b(m - 1) += a(j).toInt
              j += 1
            }
          }
        }
        RLEMasks(b.toArray, height, width)
      }
    })
  }

  def mergeRLEs(R: Array[RLEMasks], intersect: Boolean): RLEMasks = {
    val n = R.length
    if (n == 1) return R(0)
    val h = R(0).height
    val w = R(0).width
    val cnts = new ArrayBuffer[Int]()
    cnts.appendAll(R(0).counts)
    for(i <- 1 until n) {
      val B = R(i)
      require(B.height == h && B.width == w, "The height and width of the merged RLEs must" +
        " be the same")
      val acnt = cnts.toArray
      val am = cnts.length
      cnts.clear()
      var ca = uint2long(acnt(0))
      var cb = uint2long(B.counts(0))
      var (v, va, vb) = (false, false, false)
      var a = 1
      var b = 1
      var cc = 0L
      var ct = 1L

      while (ct > 0) {
        val c = Math.min(ca, cb)
        cc += c
        ct = 0
        ca -= c
        if (ca == 0 && a < am) {
          ca = uint2long(acnt(a))
          a += 1
          va = !va
        }
        ct += ca
        cb -= c
        if (cb == 0 && b < B.counts.length) {
          cb = B.get(b)
          b += 1
          vb = !vb
        }
        ct += cb
        val vp = v
        if (intersect) {
          v = va && vb
        } else {
          v = va || vb
        }
        if (v != vp || ct == 0) {
          cnts += cc.toInt
          cc = 0
        }
      }
    }
    RLEMasks(cnts.toArray, h, w)
  }
}
