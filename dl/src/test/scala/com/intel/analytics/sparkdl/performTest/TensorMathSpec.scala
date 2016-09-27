package com.intel.analytics.sparkdl.performTest

import com.intel.analytics.sparkdl.tensor._
import org.scalatest.FlatSpec
import com.intel.analytics.sparkdl.utils.RandomGenerator._

/**
  * Created by yao on 9/7/16.
  */
class TensorMathSpec extends FlatSpec {
  val Seed = 100
  RNG.setSeed(Seed)
  val sizeLarge = 4096
  val matrixLargeLeft = Tensor[Float](sizeLarge, sizeLarge).rand()
  val matrixLargeRight = Tensor[Float](sizeLarge, sizeLarge).rand()
  val vectorLarge = Tensor[Float](sizeLarge).rand()
  val sizeMid = 512
  val matrixMidLeft = Tensor[Float](sizeMid, sizeMid).rand()
  val matrixMidRight = Tensor[Float](sizeMid, sizeMid).rand()
  val vectorMid = Tensor[Float](sizeMid).rand()
  val sizeSmall = 32
  val matrixSmallLeft = Tensor[Float](sizeSmall, sizeSmall).rand()
  val matrixSmallRight = Tensor[Float](sizeSmall, sizeSmall).rand()
  val vectorSmall = Tensor[Float](sizeSmall).rand()
  val scalar = 5


  var testCase = "4096 * 4096 matrix add operation"
  TestUtils.testMathOperation(() => matrixLargeLeft.add(matrixLargeRight), testCase, 10)

  testCase = "512 * 512 matrix add operation"
  TestUtils.testMathOperation(() => matrixMidLeft.add(matrixMidRight), testCase)

  testCase = "32 * 32 matrix add operation"
  TestUtils.testMathOperation(() => matrixSmallLeft.add(matrixSmallRight), testCase)

  testCase = "4096 * 4096 matrix minus operation"
  TestUtils.testMathOperation(() => matrixLargeLeft-matrixLargeRight, testCase, 10)

  testCase = "512 * 512 matrix minus operation"
  TestUtils.testMathOperation(() => matrixMidLeft-matrixMidRight, testCase)

  testCase = "32 * 32 matrix minus operation"
  TestUtils.testMathOperation(() => matrixSmallLeft-matrixSmallRight, testCase)

  testCase = "4096 * 4096 matrix multiply operation"
  TestUtils.testMathOperation(() => matrixLargeLeft.cmul(matrixLargeRight), testCase, 10)

  testCase = "512 * 512 matrix multiply operation"
  TestUtils.testMathOperation(() => matrixMidLeft.cmul(matrixMidRight), testCase)

  testCase = "32 * 32 matrix multiply operation"
  TestUtils.testMathOperation(() => matrixSmallLeft.cmul(matrixSmallRight), testCase)

  testCase = "4096 * 4096 matrix divide operation"
  TestUtils.testMathOperation(() => matrixLargeLeft.cdiv(matrixLargeRight), testCase, 10)

  testCase = "512 * 512 matrix divide operation"
  TestUtils.testMathOperation(() => matrixMidLeft.cdiv(matrixMidRight), testCase)

  testCase = "32 * 32 matrix divide operation"
  TestUtils.testMathOperation(() => matrixSmallLeft.cdiv(matrixSmallRight), testCase)

  testCase = "4096 * 4096 matrix addmm operation"
  TestUtils.testMathOperation(() => matrixLargeLeft.addmm(matrixLargeLeft, matrixLargeRight), testCase, 10)

  testCase = "512 * 512 matrix addmm operation"
  TestUtils.testMathOperation(() => matrixMidLeft.addmm(matrixMidLeft, matrixMidRight), testCase)

  testCase = "32 * 32 matrix addmm operation"
  TestUtils.testMathOperation(() => matrixSmallLeft.addmm(matrixSmallLeft, matrixSmallRight), testCase)

  testCase = "4096 * 4096 matrix addmv operation"
  TestUtils.testMathOperation(() => vectorLarge.addmv(1, matrixLargeRight, vectorLarge), testCase, 10)

  testCase = "512 * 512 matrix addmv operation"
  TestUtils.testMathOperation(() => vectorMid.addmv(1, matrixMidRight, vectorMid), testCase)

  testCase = "32 * 32 matrix addmv operation"
  TestUtils.testMathOperation(() => vectorSmall.addmv(1, matrixSmallRight, vectorSmall), testCase)

  testCase = "4096 * 4096 matrix pow operation"
  TestUtils.testMathOperation(() => matrixLargeLeft.pow(matrixLargeRight,scalar), testCase, 10)

  testCase = "512 * 512 matrix pow operation"
  TestUtils.testMathOperation(() => matrixMidLeft.pow(matrixMidRight, scalar), testCase)

  testCase = "32 * 32 matrix pow operation"
  TestUtils.testMathOperation(() => matrixSmallLeft.pow(matrixSmallRight, scalar), testCase)

  testCase = "4096 * 4096 matrix log operation"
  TestUtils.testMathOperation(() => matrixLargeLeft.log(matrixLargeRight), testCase, 10)

  testCase = "512 * 512 matrix log operation"
  TestUtils.testMathOperation(() => matrixMidLeft.log(matrixMidRight), testCase)

  testCase = "32 * 32 matrix log operation"
  TestUtils.testMathOperation(() => matrixSmallLeft.log(matrixSmallRight), testCase)

  testCase = "4096 * 4096 matrix exp operation"
  TestUtils.testMathOperation(() => matrixLargeLeft.exp(matrixLargeRight), testCase, 10)

  testCase = "512 * 512 matrix exp operation"
  TestUtils.testMathOperation(() => matrixMidLeft.exp(matrixMidRight), testCase)

  testCase = "32 * 32 matrix exp operation"
  TestUtils.testMathOperation(() => matrixSmallLeft.exp(matrixSmallRight), testCase)

  testCase = "4096 * 4096 matrix sqrt operation"
  TestUtils.testMathOperation(() => matrixLargeLeft.sqrt(matrixLargeRight), testCase, 10)

  testCase = "512 * 512 matrix sqrt operation"
  TestUtils.testMathOperation(() => matrixMidLeft.sqrt(matrixMidRight), testCase)

  testCase = "32 * 32 matrix sqrt operation"
  TestUtils.testMathOperation(() => matrixSmallLeft.sqrt(matrixSmallRight), testCase)

  testCase = "4096 * 4096 matrix log1p operation"
  TestUtils.testMathOperation(() => matrixLargeLeft.log1p(matrixLargeRight), testCase, 10)

  testCase = "512 * 512 matrix log1p operation"
  TestUtils.testMathOperation(() => matrixMidLeft.log1p(matrixMidRight), testCase)

  testCase = "32 * 32 matrix log1p operation"
  TestUtils.testMathOperation(() => matrixSmallLeft.log1p(matrixSmallRight), testCase)
}
