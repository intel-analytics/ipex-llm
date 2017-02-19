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

import java.util

import com.intel.analytics.bigdl.models.fasterrcnn.dataset.Target
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.xml.XML

object EvalUtil {

  def cumsum(arr: Array[Int]): Array[Int] = {
    var sum = 0
    arr.map { x => sum += x; sum }
  }

  /**
   * ap = voc_ap(rec, prec, [use_07_metric])
   * Compute VOC AP given precision and recall.
   * If use_07_metric is true, uses the
   * VOC 07 11 point method (default:False)
   * @return
   */
  def vocAp(rec: Array[Double], prec: Array[Double], use_07_metric: Boolean): Double = {
    var ap = 0.0
    if (use_07_metric) {
      // 11 point metric
      var p = 0.0
      var t = 0.0
      while (t < 1.1) {
        val xgt = rec.map(x => if (x >= t) 1 else 0)
        if (xgt.sum == 0) {
          p = 0
        } else {
          p = (prec zip xgt).filter(x => x._2 == 1).map(x => x._1).max
        }
        ap = ap + p / 11.0
        t += 0.1
      }
    } else {
      // correct AP calculation
      // first append sentinel values at the end
      val mrec = new Array[Double](rec.length + 2)
      mrec(mrec.length - 1) = 1.0
      rec.copyToArray(mrec, 1)
      val mpre = new Array[Double](prec.length + 2)
      prec.copyToArray(mpre, 1)

      // compute the precision envelope
      var i = mpre.length - 1
      while (i > 0) {
        mpre(i - 1) = Math.max(mpre(i - 1), mpre(i))
        i -= 1
      }
      // to calculate area under PR curve, look for points
      // where X axis (recall) changes value
      val inds = (mrec.slice(1, mrec.length) zip mrec.slice(0, mrec.length - 1)).map(
        x => x._1 != x._2).zipWithIndex.map(x => x._2)

      // and sum (\Delta recall) * prec
      ap = inds.map(i => (mrec(i + 1) - mrec(i)) * mpre(i + 1)).sum
    }
    ap
  }


  def evalSingle(scores: Tensor[Float], bbox: Tensor[Float], gtbbox: Tensor[Float],
    gtClasses: Tensor[Float], clsInd: Int,
    ovThresh: Double = 0.5, use_07_metric: Boolean): (Array[Double], Array[Double], Double) = {
    // extract gt objects for this class
    var tp: Array[Int] = null
    var fp: Array[Int] = null
    var npos = 0
    if (scores == null || scores.nElement() == 0) {
      tp = Array[Int]()
      fp = Array[Int]()
    } else {
      val inds = (1 to gtClasses.size(1)).filter(x =>
        gtClasses.valueAt(x).toInt == clsInd + 1).toArray
      val BBGT = TensorUtil.selectMatrix(gtbbox, inds, 1)
      npos = inds.length
      val det = new Array[Boolean](gtClasses.nElement())
      // sort by confidence
      val (_, sortedIds) = scores.topk(scores.size(1), increase = false)
      val BB = TensorUtil.selectMatrix(bbox, sortedIds.storage().array().map(x => x.toInt), 1)
      // go down dets and mark TPs and FPs

      tp = new Array[Int](sortedIds.nElement())
      fp = new Array[Int](sortedIds.nElement())
      var d = 1
      while (d <= sortedIds.nElement()) {
        val bb = BB(d)
        val (ovmax, jmax) = getMaxOverlap(BBGT, bb)
        if (ovmax > ovThresh) {
          if (!det(jmax)) {
            tp(d - 1) = 1
            det(jmax) = true
          } else {
            fp(d - 1) = 1
          }
        } else {
          fp(d - 1) = 1
        }
        d += 1
      }
    }

    // compute precision recall
    fp = cumsum(fp)
    tp = cumsum(tp)
    val rec = tp.map(x => x / npos.toDouble)
    // avoid divide by zero in case the first detection matches a difficult
    // ground truth
    val prec = (tp zip (tp zip fp).map(x => x._1 + x._2)
      .map(x => Math.max(x, 2.2204460492503131e-16)))
      .map(x => x._1 / x._2)
    val ap = vocAp(rec, prec, use_07_metric)
    (rec, prec, ap)
  }

  def eval(results: Array[Array[Target]], gtBBoxes: Array[Tensor[Float]],
    gtClasses: Array[Tensor[Float]],
    clsInd: Int, ovThresh: Double = 0.5, use07Metric: Boolean = false)
  : (Array[Double], Array[Double], Double) = {
    // extract gt objects for this class
    var npos = 0
    val num = gtClasses.length
    var classRecs = Map[Int, (Tensor[Float], Array[Boolean], Array[Boolean])]()

    val imageIds = ArrayBuffer[Int]()
    val confidence: ArrayBuffer[Float] = ArrayBuffer[Float]()
    val BB: ArrayBuffer[Tensor[Float]] = ArrayBuffer[Tensor[Float]]()
    var imgInd = 0
    while (imgInd < num) {
      val gtcls = gtClasses(imgInd)
      val gtbbox = gtBBoxes(imgInd)
      val output = results(imgInd)(clsInd)
      if (gtcls.nElement > 0) {
        val selectedInds = (1 to gtcls.size(1)).filter(x =>
          gtcls.valueAt(x).toInt == clsInd + 1).toArray
        val BBGT = TensorUtil.selectMatrix(gtbbox, selectedInds, 1)
        npos += selectedInds.length
        val det = new Array[Boolean](gtcls.nElement())
        val difficult = Array.fill[Boolean](gtcls.nElement())(false)
        classRecs += (imgInd -> (BBGT, difficult, det))
      } else {
        classRecs += (imgInd -> (null, null, null))
      }
      if (output != null && output.classes.nElement() != 0) {
        imageIds.appendAll(Array.fill(output.classes.size(1))(imgInd))
        var i = 1
        while (i <= output.classes.size(1)) {
          confidence.append(output.classes.valueAt(i))
          BB.append(output.bboxes(i))
          i += 1
        }
      }
      imgInd += 1
    }
    evaluate(imageIds.toArray, classRecs, BB.toArray, confidence.toArray,
      ovThresh, npos, use07Metric)
  }

  def getMaxOverlap(BBGT: Tensor[Float], bb: Tensor[Float]): (Float, Int) = {
    if (BBGT != null && BBGT.nElement() > 0) {
      val overlaps = (1 to BBGT.size(1)).map(r => {
        val ixmin = Math.max(BBGT.valueAt(r, 1), bb.valueAt(1))
        val iymin = Math.max(BBGT.valueAt(r, 2), bb.valueAt(2))
        val ixmax = Math.min(BBGT.valueAt(r, 3), bb.valueAt(3))
        val iymax = Math.min(BBGT.valueAt(r, 4), bb.valueAt(4))
        val inter = Math.max(ixmax - ixmin + 1, 0) * Math.max(iymax - iymin + 1, 0)
        val xx = (BBGT.valueAt(r, 3) - BBGT.valueAt(r, 1) + 1) *
          (BBGT.valueAt(r, 4) - BBGT.valueAt(r, 2) + 1)
        val bbArea = (bb.valueAt(3) - bb.valueAt(1) + 1f) * (bb.valueAt(4) - bb.valueAt(2) + 1f)
        inter / (xx - inter + bbArea)
      })
      overlaps.zipWithIndex.maxBy(x => x._1)
    } else {
      (-Float.MaxValue, -1)
    }
  }

  /**
   * rec, prec, ap = voc_eval(detpath,
   * annopath,
   * imagesetfile,
   * classname,
   * [ovthresh],
   * [use_07_metric])
   * Top level function that does the PASCAL VOC evaluation.
   * @param detPath       Path to detections
   *                      detpath.format(classname) should produce the detection results file.
   * @param annoPath      Path to annotations
   *                      annopath.format(imagename) should be the xml annotations file.
   * @param imagesetFile  Text file containing the list of images, one image per line.
   * @param classname     Category name (duh)
   * @param ovThresh      Overlap threshold (default = 0.5)
   * @param use07Metric   Whether to use VOC07's 11 point AP computation
   * @return
   */
  def vocEval(detPath: String, annoPath: String, imagesetFile: String, classname: String,
    ovThresh: Double = 0.5, use07Metric: Boolean = false)
  : (Array[Double], Array[Double], Double) = {
    // assumes detections are in detpath.format(classname)
    // assumes annotations are in annopath.format(imagename)
    // assumes imagesetfile is a text file with each line an image name
    // cachedir caches the annotations in a pickle file


    // read list of images
    val imageNames = Source.fromFile(imagesetFile).getLines().map(x => x.trim).toArray

    val recs = loadAnnotations(annoPath, imageNames)

    // extract gt objects for this class
    var npos = 0

    val (classRecs, count) = getClassAnnotation(classname, recs, imageNames)
    npos += count
    // read dets
    val detfile = detPath.format(classname)
    val splitlines = Source.fromFile(detfile).getLines().map(x =>
      x.trim.split(" ")).toArray
    if (splitlines.length == 0) return (Array[Double](), Array[Double](), 0)
    val imageIds = splitlines.map(x => x(0).toInt)
    val confidence = splitlines.map(x => x(1).toFloat)
    val BB = splitlines.map(x => {
      Tensor(Storage(x.slice(2, x.length).map(z => z.toFloat)))
    })

    evaluate(imageIds, classRecs, BB, confidence, ovThresh, npos, use07Metric)
  }

  private def evaluate(imageIds: Array[Int],
    classRecs: Map[Int, (Tensor[Float], Array[Boolean], Array[Boolean])],
    BB: Array[Tensor[Float]], confidence: Array[Float],
    ovThresh: Double, npos: Int, use07Metric: Boolean)
  : (Array[Double], Array[Double], Double) = {
    // sort by confidence
    val sortedIds = if (confidence == null || confidence.length == 0) return (null, null, 0)
    else {
      confidence.zipWithIndex.sortBy(-_._1).map(_._2)
    }
    val sortedBB = sortedIds.map(id => BB(id))
    val sortedImageIds = sortedIds.map(x => imageIds(x))
    // go down dets and mark TPs and FPs
    val nd = sortedImageIds.length
    var tp = new Array[Int](nd)
    var fp = new Array[Int](nd)
    var d = 0
    while (d < nd) {
      val R = classRecs(sortedImageIds(d))
      val (ovmax, jmax) = getMaxOverlap(R._1, sortedBB(d))

      if (ovmax > ovThresh) {
        if (!R._2(jmax)) {
          if (!R._3(jmax)) {
            tp(d) = 1
            R._3(jmax) = true
          } else {
            fp(d) = 1
          }
        }
      } else {
        fp(d) = 1
      }
      d += 1
    }

    // compute precision recall
    fp = cumsum(fp)
    tp = cumsum(tp)
    val rec = tp.map(x => x / npos.toDouble)
    // avoid divide by zero in case the first detection matches a difficult
    // ground truth
    val prec = (tp zip (tp zip fp).map(x => x._1 + x._2)
      .map(x => Math.max(x, 2.2204460492503131e-16)))
      .map(x => x._1 / x._2)
    val ap = vocAp(rec, prec, use07Metric)
    (rec, prec, ap)
  }

  def getClassAnnotation(classname: String,
    recs: util.HashMap[Int, Array[Object]], imageNames: Array[String])
  : (Map[Int, (Tensor[Float], Array[Boolean], Array[Boolean])], Int) = {
    var classRecs = Map[Int, (Tensor[Float], Array[Boolean], Array[Boolean])]()
    val count = imageNames.map { imagename =>
      val R = recs.get(imagename.toInt).filter(obj => obj.name == classname)
      val bbox = Tensor[Float](R.length, 4)
      R.zip(Stream from (1)).foreach(x => bbox.update(x._2, x._1.bbox))
      val difficult = R.map(x => x.difficult)
      val det = new Array[Boolean](R.length)
      classRecs += (imagename.toInt -> (bbox, difficult, det))
      difficult.map(x => if (!x) 1 else 0).sum
    }.sum
    (classRecs, count)
  }

  var annotations: util.HashMap[Int, Array[Object]] = null

  private def loadAnnotations(annoPath: String, imageNames: Array[String])
  : util.HashMap[Int, Array[Object]] = {
    if (annotations != null) return annotations
    annotations = new util.HashMap[Int, Array[Object]]()
    imageNames.foreach(imagename =>
      annotations.put(imagename.toInt, parseRec(annoPath.format(imagename))))
    annotations
  }

  def parseRec(path: String): Array[Object] = {
    val xml = XML.loadFile(path)
    val objs = xml \\ "object"
    // Load object bounding boxes into a data frame.
    objs.map(obj => {
      // pixel indexes 1-based
      val bndbox = obj \ "bndbox"
      val x1 = (bndbox \ "xmin").text.toFloat
      val y1 = (bndbox \ "ymin").text.toFloat
      val x2 = (bndbox \ "xmax").text.toFloat
      val y2 = (bndbox \ "ymax").text.toFloat
      Object((obj \ "name").text,
        (obj \ "difficult").text == "1", Tensor(Storage(Array(x1, y1, x2, y2))))
    }).toArray
  }

  case class Object(name: String, difficult: Boolean, bbox: Tensor[Float]) {

  }

}

