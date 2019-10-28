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

package com.intel.analytics.bigdl.dataset.segmentation

import com.intel.analytics.bigdl.tensor.Tensor
import scala.collection.mutable.ArrayBuffer


abstract class SegmentationMasks extends Serializable {
  /**
   * Convert to a RLE encoded masks
   */
  def toRLE: RLEMasks

  /**
   * Get the height and width
   */
  def size: (Int, Int)
}

/**
 * A mask of regions defined by one or more polygons. The masked object(s) should have the same
 * label.
 * @param poly An array of polygons. The inner array defines one polygon, with [x1,y1,x2,y2,...]
 * @param height the height of the image
 * @param width the width of the image
 */
class PolyMasks(val poly: Array[Array[Float]], val height: Int, val width: Int) extends
  SegmentationMasks {
  override def toRLE: RLEMasks = {
    require(height > 0 && width > 0, "the height and width must > 0 for toRLE")
    MaskUtils.mergeRLEs(MaskUtils.poly2RLE(this, height, width), false)
  }

  /**
   * Get the height and width
   */
  override def size: (Int, Int) = (height, width)
}

object PolyMasks {
  def apply(poly: Array[Array[Float]], height: Int, width: Int): PolyMasks =
    new PolyMasks(poly, height, width)
}

/**
 * A mask of regions defined by RLE. The masked object(s) should have the same label.
 * This class corresponds to "uncompressed RLE" of COCO dataset.
 * RLE is a compact format for binary masks. Binary masks defines the region by assigning a boolean
 * to every pixel of the image. RLE compresses the binary masks by instead recording the runs of
 * trues and falses in the binary masks. RLE is an array of integer.
 * The first element is the length of run of falses staring from the first pixel.
 * The second element of RLE is the is the length of first run of trues.
 * e.g. binary masks: 00001110000011
 *      RLE:          ---4--3----5-2 ====> 4,3,5,2
 *
 * Also note that we don't use COCO's "compact" RLE string here because this RLE class has better
 * time & space performance.
 *
 * @param counts the RLE counts
 * @param height height of the image
 * @param width width of the image
 */
class RLEMasks(val counts: Array[Int], val height: Int, val width: Int)
  extends SegmentationMasks {
  override def toRLE: RLEMasks = this

  override def size: (Int, Int) = (height, width)

  // cached bbox value
  @transient
  lazy val bbox: (Float, Float, Float, Float) = MaskUtils.rleToOneBbox(this)

  // cached area value
  @transient
  lazy val area: Long = MaskUtils.rleArea(this)

  /**
   * Get an element in the counts. Process the overflowed int
   *
   * @param idx
   * @return
   */
  def get(idx: Int): Long = {
    MaskUtils.uint2long(counts(idx))
  }

  override def equals(obj: Any): Boolean = {
    if (obj == null) {
      return false
    }
    if (!obj.isInstanceOf[RLEMasks]) {
      return false
    }
    val other = obj.asInstanceOf[RLEMasks]
    if (this.eq(other)) {
      return true
    }

    this.counts.deep == other.counts.deep &&
      this.height == other.height &&
      this.width == other.width
  }

  override def hashCode() : Int = {
    val seed = 37
    var hash = 1
    hash = hash * seed + height
    hash = hash * seed + width
    this.counts.foreach(key => {
      hash = hash * seed + key.hashCode()
    })
    hash
  }
}

object RLEMasks {
  def apply(counts: Array[Int], height: Int, width: Int): RLEMasks =
    new RLEMasks(counts, height, width)
}


object MaskUtils {

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

  /**
   * Convert "uncompressed" RLE to "compact" RLE string of COCO
   * Implementation based on COCO's MaskApi.c
   * @param rle
   * @return RLE string
   */
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

  /**
   * Convert "compact" RLE string of COCO to "uncompressed" RLE
   * Implementation based on COCO's MaskApi.c
   * @param s the RLE string
   * @param h height of the image
   * @param w width of the image
   * @return RLE string
   */
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

  /**
   * Convert a PolyMasks to an array of RLEMasks. Note that a PolyMasks may have multiple
   * polygons. This function does not merge them. Instead, it returns the RLE for each polygon.
   * Implementation based on COCO's MaskApi.c
   * @param poly
   * @param height height of the image
   * @param width width of the image
   * @return The converted RLEs
   */
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

  /**
   * Merge multiple RLEs into one (union or intersect)
   * Implementation based on COCO's MaskApi.c
   * @param R the RLEs
   * @param intersect if true, do intersection; else find union
   * @return the merged RLE
   */
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
      var cb = B.get(0)
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

  private[segmentation] def rleArea(R: RLEMasks): Long = {
    var a = 0L
    for (j <- 1.until(R.counts.length, 2))
      a += R.get(j)
    a.toInt
  }

  /**
   * Calculate the intersection over union (IOU) of two RLEs
   * @param detection the detection RLE
   * @param groundTruth the ground truth RLE
   * @param isCrowd if groundTruth is isCrowd
   * @return IOU
   */
  def rleIOU(detection: RLEMasks, groundTruth: RLEMasks, isCrowd: Boolean): Float = {
    val gtBbox = groundTruth.bbox
    val dtBbox = detection.bbox
    require((detection.width, detection.height) == (groundTruth.width, groundTruth.height),
      "The sizes of RLEs must be the same to compute IOU")
    val iou = bboxIOU(gtBbox, dtBbox, isCrowd)

    if (iou > 0) {
      val crowd = isCrowd

      val dCnts = detection
      val gCnts = groundTruth

      var a = 1
      var b = 1

      var ca = dCnts.get(0)
      val ka = dCnts.counts.length
      var va: Boolean = false
      var vb: Boolean = false

      var cb = gCnts.get(0)
      val kb = gCnts.counts.length
      var i = 0L
      var u = 0L
      var ct = 1L

      while (ct > 0) {
        val c = math.min(ca, cb)
        if (va || vb) {
          u = u + c
          if (va && vb) i += c
        }
        ct = 0

        ca = ca - c
        if (ca == 0 && a < ka) {
          ca = dCnts.get(a)
          a += 1
          va = !va
        }
        ct += ca

        cb = cb - c
        if (cb == 0 && b < kb) {
          cb = gCnts.get(b)
          b += 1
          vb = !vb
        }
        ct += cb
      }
      if (i == 0) {
        u = 1
      } else if (crowd) {
        u = dCnts.area
      }
      i.toFloat / u
    } else {
      iou
    }
  }

  /**
   * Get the iou of two bounding boxes
   * @param gtx1 Ground truth x1
   * @param gty1 Ground truth y1
   * @param gtx2 Ground truth x2
   * @param gty2 Ground truth y2
   * @param dtx1 Detection x1
   * @param dty1 Detection y1
   * @param dtx2 Detection x2
   * @param dty2 Detection y2
   * @param isCrowd if ground truth is is crowd
   * @return
   */
  def bboxIOU(gtx1: Float, gty1: Float, gtx2: Float, gty2: Float, dtx1: Float, dty1: Float,
    dtx2: Float, dty2: Float, isCrowd: Boolean): Float = {
    val (xmin, ymin, xmax, ymax) = (gtx1, gty1, gtx2, gty2)
    val (x1, y1, x2, y2) = (dtx1, dty1, dtx2, dty2)
    val area = (xmax - xmin + 1) * (ymax - ymin + 1)
    val ixmin = Math.max(xmin, x1)
    val iymin = Math.max(ymin, y1)
    val ixmax = Math.min(xmax, x2)
    val iymax = Math.min(ymax, y2)
    val inter = Math.max(ixmax - ixmin + 1, 0) * Math.max(iymax - iymin + 1, 0)
    val detectionArea = (x2 - x1 + 1) * (y2 - y1 + 1)
    val union = if (isCrowd) detectionArea else (detectionArea + area - inter)
    inter / union
  }

  /**
   * Get the iou of two bounding boxes
   * @param groundTruth
   * @param detection
   * @param isCrowd if groundTruth is isCrowd
   * @return
   */
  def bboxIOU(groundTruth: (Float, Float, Float, Float),
    detection: (Float, Float, Float, Float), isCrowd: Boolean): Float = {
    bboxIOU(groundTruth._1, groundTruth._2, groundTruth._3, groundTruth._4,
      detection._1, detection._2, detection._3, detection._4, isCrowd)
  }

  // convert one rle to one bbox
  private[segmentation] def rleToOneBbox(rle: RLEMasks): (Float, Float, Float, Float) = {
    val m = rle.counts.length / 2 * 2

    val h = rle.height.toLong
    var xp = 0.0f
    var cc = 0L
    var xs = rle.width.toLong
    var ys = rle.height.toLong
    var ye = 0.0f
    var xe = 0.0f

    if(m == 0) {
      (0, 0, 0, 0)
    } else {
      for (j <- 0 until m) {
        cc += rle.get(j)
        val t = cc - j % 2
        val y = t % h
        val x = (t - y) / h
        if (j % 2 == 0) {
          xp = x
        } else if (xp < x) {
          ys = 0
          ye = h - 1
        }
        xs = math.min(xs, x)
        xe = math.max(xe, x)
        ys = math.min(ys, y)
        ye = math.max(ye, y)
      }
      (xs, ys, xe, ye)
    }
  }

  def polyToSingleRLE(poly: PolyMasks, height: Int, width: Int): RLEMasks = {
    val out = poly2RLE(poly, height, width)
    mergeRLEs(out, false)
  }

  // convert binary mask to rle with counts
  def binaryToRLE(binaryMask: Tensor[Float]): RLEMasks = {
    val countsBuffer = new ArrayBuffer[Int]

    val h = binaryMask.size(1)
    val w = binaryMask.size(2)
    val maskArr = binaryMask.storage().array()
    val offset = binaryMask.storageOffset() - 1

    val n = binaryMask.nElement()
    var i = 0
    var p = -1
    var c = 0
    while (i < n) {
      // the first one should be 0
      val iw = i / h
      val ih = i % h
      val ss = ih * w + iw
      if (p == -1 && maskArr(ss + offset) == 1) {
        countsBuffer.append(0)
        p = 1
        c = 1
      } else if (p == -1 && maskArr(ss + offset) == 0) {
        p = 0
        c = 1
      } else if (maskArr(ss + offset) == p) {
        c += 1
      } else {
        countsBuffer.append(c)
        c = 1
        p = maskArr(ss + offset).toInt
      }
      i += 1
    }
    countsBuffer.append(c)

    RLEMasks(countsBuffer.toArray, height = h, width = w)
  }
}
