package com.intel.analytics.sparkdl.performTest
import breeze.linalg._
import breeze.numerics._
import org.scalatest.FlatSpec

/**
  * Created by yansh on 16-9-27.
  */
class BreezeMathSpec  extends FlatSpec{
  val Seed = 100
  val sizeLarge = 4096
  val matrixLargeLeft = DenseMatrix.rand(sizeLarge, sizeLarge)
  val matrixLargeRight = DenseMatrix.rand(sizeLarge, sizeLarge)
  val vectorLarge = DenseVector.rand(sizeLarge)
  val sizeMid = 512
  val matrixMidLeft = DenseMatrix.rand(sizeMid, sizeMid)
  val matrixMidRight = DenseMatrix.rand(sizeMid, sizeMid)
  val vectorMid = DenseVector.rand(sizeMid)
  val sizeSmall = 32
  val matrixSmallLeft = DenseMatrix.rand(sizeSmall, sizeSmall)
  val matrixSmallRight = DenseMatrix.rand(sizeSmall, sizeSmall)
  val vectorSmall = DenseVector.rand(sizeSmall)
  val scalar = 5

  var testCase = " Breeze 4096 * 4096 matrix add operation"
  TestUtils.testMathOperation(() => matrixLargeLeft+matrixLargeRight, testCase, 1)

  testCase = " Breeze 512 * 512 matrix add operation"
  TestUtils.testMathOperation(() => matrixMidLeft+matrixMidRight, testCase)

  testCase = " Breeze 32 * 32 matrix add operation"
  TestUtils.testMathOperation(() => matrixSmallLeft+matrixSmallRight, testCase)

  testCase = " Breeze 4096 * 4096 matrix minus operation"
  TestUtils.testMathOperation(() => matrixLargeLeft-matrixLargeRight, testCase, 1)

  testCase = " Breeze 512 * 512 matrix minus operation"
  TestUtils.testMathOperation(() => matrixMidLeft-matrixMidRight, testCase)

  testCase = " Breeze 32 * 32 matrix minus operation"
  TestUtils.testMathOperation(() => matrixSmallLeft-matrixSmallRight, testCase)

  testCase = " Breeze 4096 * 4096 matrix multiply operation"
  TestUtils.testMathOperation(() => matrixLargeLeft*matrixLargeRight, testCase, 1)

  testCase = " Breeze 512 * 512 matrix multiply operation"
  TestUtils.testMathOperation(() => matrixMidLeft*matrixMidRight, testCase)

  testCase = " Breeze 32 * 32 matrix multiply operation"
  TestUtils.testMathOperation(() => matrixSmallLeft * matrixSmallRight, testCase)

  testCase = " Breeze 4096 * 4096 matrix divide operation"
  TestUtils.testMathOperation(() => matrixLargeLeft/matrixLargeRight, testCase, 1)

  testCase = " Breeze 512 * 512 matrix divide operation"
  TestUtils.testMathOperation(() => matrixMidLeft/matrixMidRight, testCase)

  testCase = " Breeze 32 * 32 matrix divide operation"
  TestUtils.testMathOperation(() => matrixSmallLeft/matrixSmallRight, testCase)

  /*testCase = " Breeze 4096 * 4096 matrix addmm operation"
  TestUtils.testMathOperation(() => matrixLargeLeft.addmm(matrixLargeLeft, matrixLargeRight), testCase, 10)

  testCase = " Breeze 512 * 512 matrix addmm operation"
  TestUtils.testMathOperation(() => matrixMidLeft.addmm(matrixMidLeft, matrixMidRight), testCase)

  testCase = " Breeze 32 * 32 matrix addmm operation"
  TestUtils.testMathOperation(() => matrixSmallLeft.addmm(matrixSmallLeft, matrixSmallRight), testCase)

  testCase = " Breeze 4096 * 4096 matrix addmv operation"
  TestUtils.testMathOperation(() => vectorLarge.addmv(1, matrixLargeRight, vectorLarge), testCase, 10)

  testCase = " Breeze 512 * 512 matrix addmv operation"
  TestUtils.testMathOperation(() => vectorMid.addmv(1, matrixMidRight, vectorMid), testCase)

  testCase = " Breeze 32 * 32 matrix addmv operation"
  TestUtils.testMathOperation(() => vectorSmall.addmv(1, matrixSmallRight, vectorSmall), testCase)*/

  testCase = " Breeze 4096 * 4096 matrix pow operation"
  TestUtils.testMathOperation(() => pow(matrixLargeRight,scalar), testCase, 1)

  testCase = " Breeze 512 * 512 matrix pow operation"
  TestUtils.testMathOperation(() => pow(matrixMidRight, scalar), testCase)

  testCase = " Breeze 32 * 32 matrix pow operation"
  TestUtils.testMathOperation(() => pow(matrixSmallRight, scalar), testCase)

  testCase = " Breeze 4096 * 4096 matrix log operation"
  TestUtils.testMathOperation(() => log(matrixLargeRight), testCase, 1)

  testCase = " Breeze 512 * 512 matrix log operation"
  TestUtils.testMathOperation(() => log(matrixMidRight), testCase)

  testCase = " Breeze 32 * 32 matrix log operation"
  TestUtils.testMathOperation(() => log(matrixSmallRight), testCase)

  testCase = " Breeze 4096 * 4096 matrix exp operation"
  TestUtils.testMathOperation(() => exp(matrixLargeRight), testCase, 1)

  testCase = " Breeze 512 * 512 matrix exp operation"
  TestUtils.testMathOperation(() => exp(matrixMidRight), testCase)

  testCase = " Breeze 32 * 32 matrix exp operation"
  TestUtils.testMathOperation(() => exp(matrixSmallRight), testCase)

  testCase = " Breeze 4096 * 4096 matrix sqrt operation"
  TestUtils.testMathOperation(() => sqrt(matrixLargeRight), testCase, 1)

  testCase = " Breeze 512 * 512 matrix sqrt operation"
  TestUtils.testMathOperation(() => sqrt(matrixMidRight), testCase)

  testCase = " Breeze 32 * 32 matrix sqrt operation"
  TestUtils.testMathOperation(() => sqrt(matrixSmallRight), testCase)

  testCase = " Breeze 4096 * 4096 matrix log1p operation"
  TestUtils.testMathOperation(() => log1p(matrixLargeRight), testCase, 1)

  testCase = " Breeze 512 * 512 matrix log1p operation"
  TestUtils.testMathOperation(() => log1p(matrixMidRight), testCase)

  testCase = " Breeze 32 * 32 matrix log1p operation"
  TestUtils.testMathOperation(() => log1p(matrixSmallRight), testCase)
}
