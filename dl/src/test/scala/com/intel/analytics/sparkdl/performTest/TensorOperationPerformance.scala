package com.intel.analytics.sparkdl.performTest

import com.intel.analytics.sparkdl.tensor
import org.scalatest.FlatSpec

/**
  * Created by yao on 9/7/16.
  */
class TensorOperationPerformance extends FlatSpec {
  val Seed = 100
  RNG.setSeed(Seed)
  val sizeLarge = 4096
  val matrixLargeLeft = new Tensor[Float](sizeLarge, sizeLarge).rand()
  val matrixLargeRight = torch.Tensor[Float](sizeLarge, sizeLarge).rand()
  val vectorLarge = torch.Tensor[Float](sizeLarge).rand()
  val sizeMid = 512
  val matrixMidLeft = torch.Tensor[Float](sizeMid, sizeMid).rand()
  val matrixMidRight = torch.Tensor[Float](sizeMid, sizeMid).rand()
  val vectorMid = torch.Tensor[Float](sizeMid).rand()
  val sizeSmall = 32
  val matrixSmallLeft = torch.Tensor[Float](sizeSmall, sizeSmall).rand()
  val matrixSmallRight = torch.Tensor[Float](sizeSmall, sizeSmall).rand()
  val vectorSmall = torch.Tensor[Float](sizeSmall).rand()
  val scalar = 128


  var testCase = "4096 * 4096 matrix add operation"
  TestUtils.testMathOperation(() => matrixLargeLeft.add(matrixLargeRight), testCase, 10)

  testCase = "512 * 512 matrix add operation"
  TestUtils.testMathOperation(() => matrixMidLeft.add(matrixMidRight), testCase)

  testCase = "32 * 32 matrix add operation"
  TestUtils.testMathOperation(() => matrixSmallLeft.add(matrixSmallRight), testCase)

  testCase = "4096 * 4096 matrix minus operation"
  TestUtils.testMathOperation(() => matrixLargeLeft.sub(matrixLargeRight), testCase, 10)

  testCase = "512 * 512 matrix minus operation"
  TestUtils.testMathOperation(() => matrixMidLeft.sub(matrixMidRight), testCase)

  testCase = "32 * 32 matrix minus operation"
  TestUtils.testMathOperation(() => matrixSmallLeft.sub(matrixSmallRight), testCase)

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
  TestUtils.testMathOperation(() => vectorLarge.addmv(scalar, matrixLargeRight, vectorLarge), testCase, 10)

  testCase = "512 * 512 matrix addmv operation"
  TestUtils.testMathOperation(() => vectorMid.addmv(scalar, matrixMidRight, vectorMid), testCase)

  testCase = "32 * 32 matrix addmv operation"
  TestUtils.testMathOperation(() => vectorSmall.addmv(scalar, matrixSmallRight, vectorSmall), testCase)
}
