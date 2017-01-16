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

import java.io.{File, PrintWriter}
import java.nio.file.Paths

import com.intel.analytics.bigdl.models.fasterrcnn.dataset.Target
import com.intel.analytics.bigdl.models.fasterrcnn.utils.PascalVocEvaluator._
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.commons.lang3.SerializationUtils
import org.apache.log4j.Logger
import scopt.OptionParser

import scala.collection.mutable.ArrayBuffer


/**
 *
 * @param imageSet voc_2007_test, voc_2007_train, etc
 */
class PascalVocEvaluator(imageSet: String) extends Serializable {


  private val year: String = imageSet.split("_")(1)
  private val setName: String = imageSet.split("_")(2)

  // The PASCAL VOC metric changed in 2010
  def use07metric: Boolean = year == "2007"


  /**
   * evaluate the results of detections to get the mean map for each class
   * @param results   the list of results, each inner array is the detection results
   *                  for each class, and outer array contains output for each image
   * @param gtBboxes  the array of ground truth boxes for all images
   * @param gtClasses the array of ground truth labels for all images,
   *                  the lenth should be equal to the number of bboxes.
   */
  private def evaluateDetections(results: Array[Array[Target]],
    gtBboxes: Array[Tensor[Float]],
    gtClasses: Array[Tensor[Float]]): Unit = {
    eval(results, gtBboxes, gtClasses)
  }

  def evaluateDetections(results: Array[(Array[Target], Tensor[Float],
    Tensor[Float], String)], outPath: String): Unit = {
    val allResults: Array[Array[Target]] = {
      val out = new Array[Array[Target]](results.length)
      var x = 0
      while (x < results.length) {
        out(x) = new Array[Target](classes.length)
        x += 1
      }
      out
    }
    val allGtBoxes: Array[Tensor[Float]] = new Array[Tensor[Float]](results.length)
    val allLabels = new Array[Tensor[Float]](results.length)
    results.zipWithIndex.foreach(x => {
      allResults(x._2) = x._1._1
      allGtBoxes(x._2) = x._1._2
      allLabels(x._2) = x._1._3
    })
    evaluateDetections(allResults, allGtBoxes, allLabels)
    writeVocResultsFile(results.sortBy(_._4).map(x => (x._4, x._1)), outPath)
  }


  def evaluateSingle(results: Array[Target],
    bbox: Tensor[Float], gtClasses: Tensor[Float], path: String = ""): Unit = {
    classes.zipWithIndex.foreach {
      case (cls, clsInd) =>
        if (cls != "__background__") {
          val (_, _, ap) = EvalUtil.evalSingle(results(clsInd).classes, results(clsInd).bboxes,
            bbox, gtClasses, clsInd, ovThresh = 0.5, use07metric)
          def print(): Unit = {
            var i = 1
            while (i <= gtClasses.nElement()) {
              if (clsInd + 1 == gtClasses.valueAt(i)) {
                logger.info(s"$path -- $cls = ${ "%.4f".format(ap) }")
                return
              }
              i += 1
            }
          }
          print()
        }
    }
  }

  private def printInfo(aps: ArrayBuffer[Double], output: ArrayBuffer[(String, Double)]): Unit = {
    logger.info(s"Mean AP = ${ "%.4f".format(aps.sum / aps.length) }")
    output.append(("Mean AP", aps.sum / aps.length))
    logger.info("~~~~~~~~")
    logger.info("Results:")
    aps.foreach(ap => logger.info(s"${ "%.3f".format(ap) }"))
    logger.info(s"${ "%.3f".format(aps.sum / aps.length) }")
    logger.info("~~~~~~~~")
  }

  private def eval(results: Array[Array[Target]], gtBboxes: Array[Tensor[Float]],
    gtClasses: Array[Tensor[Float]]): Array[(String, Double)] = {
    val aps = ArrayBuffer[Double]()
    logger.info("VOC07 metric ? " + (if (use07metric) "yes" else "No"))
    val output = ArrayBuffer[(String, Double)]()
    var i = 0
    while (i < classes.length) {
      val cls = classes(i)
      if (cls != "__background__") {
        val (_, _, ap) = EvalUtil.eval(results, gtBboxes, gtClasses, i,
          ovThresh = 0.5, use07Metric = use07metric)
        aps.append(ap)
        logger.info(s"AP for $cls = ${ "%.4f".format(ap) }")
        output.append((cls, ap))
      }
      i += 1
    }
    printInfo(aps, output)
    output.toArray
  }

  def eval(devkitPath: String, resultFolder: String): Array[(String, Double)] = {
    val annopath = s"$devkitPath/VOC$year/Annotations/%s.xml"
    val imagesetfile = s"$devkitPath/VOC$year/ImageSets/Main" +
      s"/${ setName }.txt"
    val aps = ArrayBuffer[Double]()
    println("VOC07 metric ? " + (if (use07metric) "yes" else "No"))
    val results = ArrayBuffer[(String, Double)]()
    var i = 0
    while (i < classes.length) {
      val cls = classes(i)
      if (cls != "__background__") {
        val filename = getVocResultsFileTemplate(new File(resultFolder), cls)
        val (_, _, ap) = EvalUtil.vocEval(filename, annopath, imagesetfile, cls,
          ovThresh = 0.5, use07metric)
        aps.append(ap)
        logger.info(s"AP for $cls = ${ "%.4f".format(ap) }")
        results.append((cls, ap))
      }
      i += 1
    }
    printInfo(aps, results)
    cleanupCache()
    results.toArray
  }

  private def cleanupCache(): Unit = {
    EvalUtil.annotations = null
  }

  private def extractImgIdFromPath(path: String): String = {
    if (path.contains("/")) path.substring(path.lastIndexOf("/") + 1, path.lastIndexOf("."))
    else path
  }

  private def writeVocResultsFile(allBoxes: Array[(String, Array[Target])],
    outPath: String) = {
    var clsInd = 0
    while (clsInd < classes.length) {
      val cls = classes(clsInd)
      if (cls != "__background__") {
        val filename = getVocResultsFileTemplate(new File(outPath), cls)
        val of = new PrintWriter(new java.io.File(filename))
        allBoxes.zipWithIndex.foreach {
          case ((imInd, out), index) =>
            val dets = out(clsInd)
            if (dets != null && dets.classes.nElement() > 0) {
              // the VOCdevkit expects 1-based indices
              var k = 1
              while (k <= dets.classes.size(1)) {
                of.write("%s %.3f %.1f %.1f %.1f %.1f\n".format(
                  extractImgIdFromPath(imInd), dets.classes.valueAt(k),
                  dets.bboxes.valueAt(k, 1) + 1, dets.bboxes.valueAt(k, 2) + 1,
                  dets.bboxes.valueAt(k, 3) + 1, dets.bboxes.valueAt(k, 4) + 1
                ))
                k += 1
              }
            }
        }
        of.close()
        logger.info(s"writing $cls VOC results file $filename")
      }
      clsInd += 1
    }
  }

  /**
   * VOCdevkit / results / VOC2007 / Main /< comp_id > _det_test_aeroplane.txt
   */
  def getVocResultsFileTemplate(root: File, cls: String): String = {
    if (!root.exists()) {
      root.mkdirs()
    }
    Paths.get(root.toString, s"comp4_det_${ setName }_${ cls }.txt").toString
  }

  def cloneEvaluator(): PascalVocEvaluator = {
    SerializationUtils.clone(this)
  }
}

object PascalVocEvaluator {
  val logger = Logger.getLogger(getClass)
  val classes = Array[String](
    "__background__", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
  )

  case class EvaluateParam(
    devkitPath: String = "",
    resultPath: String = "",
    imageSet: String = "")

  val parser = new OptionParser[EvaluateParam]("Spark-DL Faster-RCNN Evaluate") {
    head("Spark-DL Faster-RCNN Evaluate")
    opt[String]('r', "folder")
      .text("where you put the result data")
      .action((x, c) => c.copy(resultPath = x))
      .required()
    opt[String]('f', "folder")
      .text("where you put the PascolVoc data")
      .action((x, c) => c.copy(devkitPath = x))
      .required()
    opt[String]('i', "imageset")
      .text("pascal voc imageset")
      .action((x, c) => c.copy(imageSet = x))
      .required()
  }

  def main(args: Array[String]): Unit = {
    val params = parser.parse(args, EvaluateParam()).get
    val evaluator = new PascalVocEvaluator(params.imageSet)
    evaluator.eval(params.devkitPath, params.resultPath)
  }
}
