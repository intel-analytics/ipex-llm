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

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.transform.vision.image.RoiImageInfo
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}
import scala.collection.mutable.ArrayBuffer

@com.intel.analytics.bigdl.tags.Parallel
class ValidationSpec extends FlatSpec with Matchers {
  "MAPUtil" should "be correct for find top k" in {
    val arr = Array(
      Array(1f, 2f, 3f, 4f, 5f), // 0
      Array(0f, 0f, 4f, 6f, 7f), // 1
      Array(6f, 4f, 1f, 5f, 2f), // 2
      Array(3f, 5f, 0f, 1f, 9f), // 3
      Array(1f, 2f, 3f, 2f, 5f), // 4
      Array(0f, 0f, 4f, 9f, 7f), // 5
      Array(6f, 4f, 1f, 8f, 2f), // 6
      Array(3f, 5f, 0f, 3f, 9f), // 7
      Array(6f, 4f, 1f, 7f, 2f)  // 8
    )
    val result = MAPUtil.findTopK(16, arr, 3)
    val test = Array((5, 9f), (6, 8f), (8, 7f), (1, 6f), (2, 5f), (0, 4f), (7, 3f), (4, 2f),
      (3, 1f))
    result should be(test)

    val result2 = MAPUtil.findTopK(5, arr, 3)
    val test2 = Array((5, 9f), (6, 8f), (8, 7f), (1, 6f), (2, 5f))
    result2 should be(test2)
  }

  "MAPValidationResult" should "function well" in {
    val confidence = Array(
      Array(1f, 2f, 3f, 4f, 5f), // 0
      Array(0f, 0f, 4f, 6f, 7f), // 1
      Array(6f, 4f, 1f, 5f, 2f), // 2
      Array(3f, 5f, 0f, 1f, 9.1f), // 3
      Array(1f, 3f, 3f, 2f, 5f), // 4
      Array(1f, 0f, 4f, 9f, 7f), // 5
      Array(6f, 4f, 1f, 8f, 2.1f), // 6
      Array(3f, 5f, 0f, 3f, 9f), // 7
      Array(6f, 4f, 1f, 7f, 2.1f)  // 8
    )
    val gt = Array(
      1f, // 0
      0f, // 1
      4f, // 2
      4f, // 3
      3f, // 4
      3f, // 5
      3f, // 6
      2f, // 7
      3f  // 8
    )

    def mkArray(data: (Float, Boolean)*): ArrayBuffer[(Float, Boolean)] = {
      val ret = new ArrayBuffer[(Float, Boolean)]()
      ret.appendAll(data)
      ret
    }
    val result = new MAPValidationResult(5, 8,
      Array(
        mkArray((1f, false), (0f, true), (6f, false), (3f, false), (1f, false), (1f, false), (6f,
          false), (3f, false), (6f, false)),
        mkArray((2f, true), (0f, false), (4f, false), (5f, false), (3f, false), (0f, false), (4f,
          false), (5f, false), (4f, false)),
        mkArray((3f, false), (4f, false), (1f, false), (0f, false), (3f, false), (4f, false),
          (1f, false), (0f, true), (1f, false)),
        mkArray((4f, false), (6f, false), (5f, false), (1f, false), (2f, true), (9f, true), (8f,
          true), (3f, false), (7f, true)),
        mkArray((5f, false), (7f, false), (2f, true), (9.1f, true), (5f, false), (7f, false),
          (2.1f, false), (9f, false), (2.1f, false))
      ), Array(1, 1, 1, 4, 2))
    val ap1 = result.calculateClassAP(0)
    ap1 should be (0f)
    val ap2 = result.calculateClassAP(1)
    ap2 should be (1f/7f)
    val ap3 = result.calculateClassAP(2)
    ap3 should be (0f)
    val ap4 = result.calculateClassAP(3)
    ap4 should be (0.875f)
    val ap5 = result.calculateClassAP(4)
    ap5 should be (0.5f)

    result.result()._1 should be(0.303571429f +- 1e-5f)
  }

  "MAPValidationResult" should "merge well" in {
    def predictForClass1: Array[ArrayBuffer[(Float, Boolean)]] = (1 to 5).map(i => {
      val p = new ArrayBuffer[(Float, Boolean)]
      for (j <- 101 to 200) {
        p.append((j.toFloat, true))
      }
      p
    }).toArray

    def predictForClass2: Array[ArrayBuffer[(Float, Boolean)]] = (1 to 5).map(i => {
      val p = new ArrayBuffer[(Float, Boolean)]
      for (j <- 201 to 210) {
        p.append((j.toFloat, true))
      }
      p
    }).toArray

    def predictForClass3: Array[ArrayBuffer[(Float, Boolean)]] = (1 to 5).map(i => {
      val p = new ArrayBuffer[(Float, Boolean)]
      for (j <- 51 to 100) {
        p.append((j.toFloat, true))
      }
      for (j <- 211 to 260) {
        p.append((j.toFloat, true))
      }
      p
    }).toArray

    {
      val vr1 = new MAPValidationResult(5, -1, predictForClass1, Array(1, 2, 3, 4, 5))
      val vr2 = new MAPValidationResult(5, -1, predictForClass2, Array(6, 7, 8, 9, 10))
      val vr3 = new MAPValidationResult(5, -1, predictForClass3, Array(3, 2, 1, 0, 2))

      val tmpv = vr1 + vr2
      tmpv.asInstanceOf[MAPValidationResult].predictForClass.foreach(p => {
        p.zip(101 to 210).foreach(p => p._1._1 should be (p._2.toFloat))
      })
      vr1 + vr3
      vr1.asInstanceOf[MAPValidationResult].predictForClass.foreach(p => {
        p.sortBy(_._1).zip(51 to 260).foreach(p => p._1._1 should be (p._2.toFloat))
      })
    }

    {
      val vr1 = new MAPValidationResult(5, 150, predictForClass1, Array(1, 2, 3, 4, 5))
      val vr2 = new MAPValidationResult(5, 150, predictForClass2, Array(6, 7, 8, 9, 10))
      val vr3 = new MAPValidationResult(5, 150, predictForClass3, Array(3, 2, 1, 0, 2))

      val tmpv = vr1 + vr2
      tmpv.asInstanceOf[MAPValidationResult].predictForClass.foreach(p => {
        p.sortBy(_._1).zip(101 to 210).foreach(p => p._1._1 should be (p._2.toFloat))
      })
      vr1 + vr3
      vr1.asInstanceOf[MAPValidationResult].predictForClass.foreach(p => {
        p.sortBy(_._1).zip(111 to 260).foreach(p => p._1._1 should be (p._2.toFloat))
      })
      vr1.gtCntForClass.zip(Array(10, 11, 12, 13, 17)).foreach(p => p._1 should be (p._2))
    }

  }

  "MeanAveragePrecision" should "be correct on 1d tensor" in {
    implicit val numeric = TensorNumeric.NumericFloat
    val output = Tensor[Float](
      T(
        T(6f, 4f, 1f, 5f, 2f), // 2
        T(3f, 5f, 0f, 1f, 9.1f), // 3
        T(1f, 3f, 3f, 2f, 5f), // 4
        T(1f, 0f, 4f, 9f, 7f), // 5
        T(6f, 4f, 1f, 8f, 2.1f), // 6
        T(3f, 5f, 0f, 3f, 9f), // 7
        T(6f, 4f, 1f, 7f, 2.1f)  // 8
     ))

    val target = Tensor[Float](
      T(T(
        4f, // 2
        4f, // 3
        3f, // 4
        3f, // 5
        3f, // 6
        2f, // 7
        3f  // 8
      )))

    val r0 = new MeanAveragePrecision(8, 5).apply(output, target)
    val r1 = new MeanAveragePrecision(8, 5).apply(Tensor[Float](T(1f, 2f, 3f, 4f, 5f)),
      Tensor[Float](T(1f)))
    val r2 = new MeanAveragePrecision(8, 5).apply(Tensor[Float](T(0f, 0f, 4f, 6f, 7f)),
      Tensor[Float](T(0f)))
    (r0 + r1 + r2).result()._1 should be(0.303571429f +- 1e-5f)
  }

  "MeanAveragePrecision" should "be correct on 2d tensor" in {
    implicit val numeric = TensorNumeric.NumericFloat
    val output = Tensor[Float](
      T(
        T(1f, 2f, 3f, 4f, 5f), // 0
        T(0f, 0f, 4f, 6f, 7f), // 1
        T(6f, 4f, 1f, 5f, 2f), // 2
        T(3f, 5f, 0f, 1f, 9.1f), // 3
        T(1f, 3f, 3f, 2f, 5f), // 4
        T(1f, 0f, 4f, 9f, 7f), // 5
        T(6f, 4f, 1f, 8f, 2.1f), // 6
        T(3f, 5f, 0f, 3f, 9f), // 7
        T(6f, 4f, 1f, 7f, 2.1f)  // 8
      ))

    val target = Tensor[Float](
      T(T(
        1f, // 0
        0f, // 1
        4f, // 2
        4f, // 3
        3f, // 4
        3f, // 5
        3f, // 6
        2f, // 7
        3f  // 8
      )))
    val v = new MeanAveragePrecision(8, 5)
    val result = v(output, target)
    result.result()._1 should be(0.303571429f +- 1e-5f)
  }

  "MeanAveragePrecisionObjectDetection" should "be correct" in {
    implicit val numeric = TensorNumeric.NumericFloat
    val output = Tensor[Float](
      T(
        T(8f,
          // label score bbox
          0, 1, 110, 90, 210, 190,
          0, 2, 310, 110, 410, 210,
          0, 4, 320, 290, 420, 390,
          0, 3, 210, 310, 290, 410,
          1, 1, 1110, 1090, 1210, 1190,
          1, 3, 1310, 1110, 1410, 1210,
          1, 4, 1320, 1290, 1420, 1390,
          1, 2, 1210, 1310, 1290, 1410
        )
      ))

    val target = T(
        T()
          .update(RoiImageInfo.ISCROWD, Tensor[Float](T(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
          .update(RoiImageInfo.CLASSES, Tensor[Float](T(0, 0, 0, 0, 0, 1, 1, 1, 1, 1)))
          .update(RoiImageInfo.BBOXES, Tensor[Float](T(
              T(100, 100, 200, 200),
              T(300, 100, 400, 200),
              T(100, 300, 200, 400),
              T(300, 300, 400, 400),
              T(210, 210, 230, 290),
              T(1100, 1100, 1200, 1200),
              T(1300, 1100, 1400, 1200),
              T(1100, 1300, 1200, 1400),
              T(1300, 1300, 1400, 1400),
              T(1210, 1210, 1230, 1290)
            ))
          )
      )

    val v = new MeanAveragePrecisionObjectDetection(3)
    val result = v(output, target)
    // 0.5f and 0.55f
    result.result()._1 should be(0.35f +- 1e-5f)

    val outputTable = T(
      T()
        .update(RoiImageInfo.CLASSES, Tensor[Float](T(0, 0, 0, 0, 1, 1, 1, 1)))
        .update(RoiImageInfo.BBOXES, Tensor[Float](T(
          T(110, 90, 210, 190),
          T(310, 110, 410, 210),
          T(320, 290, 420, 390),
          T(210, 310, 290, 410),
          T(1110, 1090, 1210, 1190),
          T(1310, 1110, 1410, 1210),
          T(1320, 1290, 1420, 1390),
          T(1210, 1310, 1290, 1410)
        ))
        )
        .update(RoiImageInfo.SCORES, Tensor[Float](T(1, 2, 4, 3, 1, 3, 4, 2)))
    )
    val v2 = new MeanAveragePrecisionObjectDetection(3)
    val result2 = v2(outputTable, target)
    // 0.5f and 0.55f
    result2.result()._1 should be(0.35f +- 1e-5f)
  }

  "MeanAveragePrecisionObjectDetection" should "be correct on empty detections" in {
    val target = T(
      T()
        .update(RoiImageInfo.ISCROWD, Tensor[Float](T(0, 0, 0, 0, 0)))
        .update(RoiImageInfo.CLASSES, Tensor[Float](T(0, 0, 0, 0, 0)))
        .update(RoiImageInfo.BBOXES, Tensor[Float](T(
          T(100, 100, 200, 200),
          T(300, 100, 400, 200),
          T(100, 300, 200, 400),
          T(300, 300, 400, 400),
          T(210, 210, 230, 290)
        ))
        )
    )
    val outputTable = T(T())
    val v = new MeanAveragePrecisionObjectDetection[Float](3)
    val result = v(outputTable, target)
    result.result()._1 should be(0f)
  }

  "MeanAveragePrecisionObjectDetection" should "be correct on empty targets" in {
    val target = T(
      T()
        .update(RoiImageInfo.ISCROWD, Tensor[Float](T(0, 0, 0, 0, 0)))
        .update(RoiImageInfo.CLASSES, Tensor[Float](T(0, 0, 0, 0, 0)))
        .update(RoiImageInfo.BBOXES, Tensor[Float](T(
          T(100, 100, 200, 200),
          T(300, 100, 400, 200),
          T(100, 300, 200, 400),
          T(300, 300, 400, 400),
          T(210, 210, 230, 290)
        ))
        ),
      // Empty target
      T()
    )
    val outputTable = T(
      T()
        .update(RoiImageInfo.CLASSES, Tensor[Float](T(0, 0, 0, 0)))
        .update(RoiImageInfo.BBOXES, Tensor[Float](T(
          T(110, 90, 210, 190),
          T(310, 110, 410, 210),
          T(320, 290, 420, 390),
          T(210, 310, 290, 410)
        ))
        )
        .update(RoiImageInfo.SCORES, Tensor[Float](T(1, 2, 9, 7))),
      T()
        .update(RoiImageInfo.CLASSES, Tensor[Float](T(0, 0, 0, 0)))
        .update(RoiImageInfo.BBOXES, Tensor[Float](T(
          T(1110, 1090, 1210, 1190),
          T(1310, 1110, 1410, 1210),
          T(1320, 1290, 1420, 1390),
          T(1210, 1310, 1290, 1410)
        ))
        )
        .update(RoiImageInfo.SCORES, Tensor[Float](T(0, 5, 4, 8)))
    )
    val v = new MeanAveragePrecisionObjectDetection[Float](3)
    val result = v(outputTable, target)
    result.result()._1 should be(0.123809524f +- 0.00000001f)
  }

  "treeNN accuracy" should "be correct on 2d tensor" in {
    val output = Tensor[Double](
      T(
        T(0.0, 0.0, 0.1, 0.0),
        T(3.0, 7.0, 0.0, 1.0),
        T(0.0, 1.0, 0.0, 0.0)))

    val target = Tensor[Double](
      T(T(3.0)))

    val validation = new TreeNNAccuracy[Double]()
    val result = validation(output, target)
    val test = new AccuracyResult(1, 1)
    result should be(test)
  }

  "treeNN accuracy" should "be correct on 3d tensor" in {
    val output = Tensor[Double](
      T(
        T(
          T(0.0, 0.0, 0.1, 0.0),
          T(3.0, 7.0, 0.0, 1.0),
          T(0.0, 1.0, 0.0, 0.0)),
        T(
          T(0.0, 0.1, 0.0, 0.0),
          T(3.0, 7.0, 0.0, 1.0),
          T(0.0, 1.0, 0.0, 0.0)),
        T(
          T(0.0, 0.0, 0.0, 0.1),
          T(3.0, 7.0, 0.0, 1.0),
          T(0.0, 1.0, 0.0, 0.0)),
        T(
          T(0.0, 0.0, 0.0, 1.0),
          T(3.0, 0.0, 8.0, 1.0),
          T(0.0, 1.0, 0.0, 0.0))))

    val target = Tensor[Double](
      T(
        T(3.0, 0.0, 0.1, 1.0),
        T(2.0, 0.0, 0.1, 1.0),
        T(3.0, 7.0, 0.0, 1.0),
        T(4.0, 1.0, 0.0, 0.0)))

    val validation = new TreeNNAccuracy[Double]()
    val result = validation(output, target)
    val test = new AccuracyResult(3, 4)
    result should be(test)
  }

  "top1 accuracy" should "be correct on 2d tensor" in {
    val output = Tensor(Storage(Array[Double](
      0, 0, 0, 1,
      0, 1, 0, 0,
      1, 0, 0, 0,
      0, 0, 1, 0,
      1, 0, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
      0, 1, 0, 0
    )), 1, Array(8, 4))

    val target = Tensor(Storage(Array[Double](
      4,
      2,
      1,
      3,
      2,
      2,
      2,
      4
    )))

    val validation = new Top1Accuracy[Double]()
    val result = validation(output, target)
    val test = new AccuracyResult(4, 8)
    result should be(test)
  }

  "top1 accuracy" should "shouldn't change output" in {
    val output = Tensor(Storage(Array[Double](
      0.1,
      0.2,
      0.8,
      0,
      0.9,
      0,
      0,
      0
    )), 1, Array(8, 1))
    val cloneOutput = output.clone()

    val target = Tensor(Storage(Array[Double](
      1,
      0,
      1,
      1,
      0,
      0,
      1,
      1
    )))

    val validation = new Top1Accuracy[Double]()
    validation(output, target)
    output should be (cloneOutput)
  }


  "top1 accuracy" should "be correct on 2d tensor for binary inputs" in {
    val output = Tensor(Storage(Array[Double](
      0,
      0,
      1,
      0,
      1,
      0,
      0,
      0
    )), 1, Array(8, 1))

    val target = Tensor(Storage(Array[Double](
      1,
      0,
      1,
      1,
      0,
      0,
      1,
      1
    )))

    val validation = new Top1Accuracy[Double]()
    val result = validation(output, target)
    val test = new AccuracyResult(3, 8)
    result should be(test)
  }

  it should "be correct on 1d tensor" in {
    val output = Tensor(Storage(Array[Double](
      0, 0, 0, 1
    )))

    val target1 = Tensor(Storage(Array[Double](
      4
    )))

    val target2 = Tensor(Storage(Array[Double](
      2
    )))

    val validation = new Top1Accuracy[Double]()
    val result1 = validation(output, target1)
    val test1 = new AccuracyResult(1, 1)
    result1 should be(test1)

    val result2 = validation(output, target2)
    val test2 = new AccuracyResult(0, 1)
    result2 should be(test2)
  }

  "Top5 accuracy" should "be correct on 2d tensor" in {
    val output = Tensor(Storage(Array[Double](
      0, 0, 8, 1, 2, 0, 0, 0,
      0, 1, 0, 0, 2, 3, 4, 6,
      1, 0, 0, 0.6, 0.1, 0.2, 0.3, 0.4,
      0, 0, 1, 0, 0.5, 1.5, 2, 0,
      1, 0, 0, 6, 2, 3, 4, 5,
      0, 0, 1, 0, 1, 1, 1, 1,
      0, 0, 0, 1, 1, 2, 3, 4,
      0, 1, 0, 0, 2, 4, 3, 2
    )), 1, Array(8, 8))

    val target = Tensor(Storage(Array[Double](
      4,
      2,
      1,
      3,
      2,
      2,
      2,
      4
    )))

    val validation = new Top5Accuracy[Double]()
    val result = validation(output, target)
    val test = new AccuracyResult(4, 8)
    result should be(test)
  }

  it should "be correct on 1d tensor" in {
    val output = Tensor(Storage(Array[Double](
      0.1, 0.2, 0.6, 0.01, 0.005, 0.005, 0.05, 0.03
    )))

    val target1 = Tensor(Storage(Array[Double](
      2
    )))

    val target2 = Tensor(Storage(Array[Double](
      5
    )))

    val target3 = Tensor(Storage(Array[Double](
      3
    )))

    val target4 = Tensor(Storage(Array[Double](
      7
    )))

    val validation = new Top5Accuracy[Double]()
    val result1 = validation(output, target1)
    val test1 = new AccuracyResult(1, 1)
    result1 should be(test1)

    val result2 = validation(output, target2)
    val test2 = new AccuracyResult(0, 1)
    result2 should be(test2)

    val result3 = validation(output, target3)
    val test3 = new AccuracyResult(1, 1)
    result3 should be(test3)

    val result4 = validation(output, target4)
    val test4 = new AccuracyResult(1, 1)
    result4 should be(test4)
  }

  "MAE" should "be correct on 2d tensor" in {
    val output = Tensor(Storage(Array[Double](
      0.1, 0.15, 0.7, 0.1, 0.05,
      0.1, 0.6, 0.1, 0.1, 0.1,
      0.8, 0.05, 0.05, 0.05, 0.05,
      0.1, 0.05, 0.7, 0.1, 0.05
    )), 1, Array(4, 5))

    val target = Tensor(Storage(Array[Double](
      4,
      3,
      4,
      2
    )))

    val validation = new MAE[Double]()
    val result = validation(output, target)
    val test = new LossResult(1.5f, 1)
    result should be(test)
  }

  "top1 accuracy" should "be correct on 2d tensor with diff size of output and target" in {
    val output = Tensor(Storage(Array[Double](
      0, 0, 0, 1,
      0, 1, 0, 0,
      1, 0, 0, 0,
      0, 0, 1, 0,
      1, 0, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
      0, 1, 0, 0
    )), 1, Array(8, 4))

    val target = Tensor(Storage(Array[Double](
      4,
      2,
      1,
      3,
      2,
      2
    )))

    val validation = new Top1Accuracy[Double]()
    val result = validation(output, target)
    val test = new AccuracyResult(4, 6)
    result should be(test)
  }

  "Top5 accuracy" should "be correct on 2d tensor with diff size of output and target" in {
    val output = Tensor(Storage(Array[Double](
      0, 0, 8, 1, 2, 0, 0, 0,
      0, 1, 0, 0, 2, 3, 4, 6,
      1, 0, 0, 0.6, 0.1, 0.2, 0.3, 0.4,
      0, 0, 1, 0, 0.5, 1.5, 2, 0,
      1, 0, 0, 6, 2, 3, 4, 5,
      0, 0, 1, 0, 1, 1, 1, 1,
      0, 0, 0, 1, 1, 2, 3, 4,
      0, 1, 0, 0, 2, 4, 3, 2
    )), 1, Array(8, 8))

    val target = Tensor(Storage(Array[Double](
      4,
      2,
      1,
      3,
      2,
      2
    )))

    val validation = new Top5Accuracy[Double]()
    val result = validation(output, target)
    val test = new AccuracyResult(4, 6)
    result should be(test)
  }

  "HR@10" should "works fine" in {
    val o = Tensor[Float].range(1, 1000, 1).apply1(_ / 1000)
    val t = Tensor[Float](1000).zero
    t.setValue(1000, 1)
    val hr = new HitRatio[Float](negNum = 999)
    val r1 = hr.apply(o, t).result()
    r1._1 should be (1.0)

    o.setValue(1000, 0.9988f)
    val r2 = hr.apply(o, t).result()
    r2._1 should be (1.0)

    o.setValue(1000, 0.9888f)
    val r3 = hr.apply(o, t).result()
    r3._1 should be (0.0f)
  }

  "ndcg" should "works fine" in {
    val o = Tensor[Float].range(1, 1000, 1).apply1(_ / 1000)
    val t = Tensor[Float](1000).zero
    t.setValue(1000, 1)
    val ndcg = new NDCG[Float](negNum = 999)
    val r1 = ndcg.apply(o, t).result()
    r1._1 should be (1.0)

    o.setValue(1000, 0.9988f)
    val r2 = ndcg.apply(o, t).result()
    r2._1 should be (0.63092977f)

    o.setValue(1000, 0.9888f)
    val r3 = ndcg.apply(o, t).result()
    r3._1 should be (0.0f)
  }

  "CongituousResult" should "works fine" in {
    val cr1 = new ContiguousResult(0.2f, 2, "HR@10")
    val cr2 = new ContiguousResult(0.1f, 1, "HR@10")
    val result = cr1 + cr2
    result.result()._1 should be (0.1f)
    result.result()._2 should be (3)
  }

  "precision recall auc" should "work correctly" in {
    val output = Tensor(Storage(Array[Float](
      0.1f, 0.4f, 0.35f, 0.8f
    )))

    val target = Tensor(Storage(Array[Float](
      0, 0, 1, 1
    )))

    val validation = new PrecisionRecallAUC[Float]()
    val result = validation(output, target)

    val auc = result.result()._1
    val num = result.result()._2

    auc should be (0.7916667f)
    num should be (4)
  }

  "precision recall auc with empty tensor" should "work correctly" in {
    val output = Tensor[Float]()
    val target = Tensor[Float]()

    val validation = new PrecisionRecallAUC[Float]()

    val thrown = intercept[IllegalArgumentException] {
      validation(output, target)
    }
  }
}
