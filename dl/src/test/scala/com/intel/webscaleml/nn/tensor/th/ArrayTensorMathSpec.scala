package com.intel.webscaleml.nn.tensor.th

import com.intel.webscaleml.nn.tensor.{Tensor, torch, DenseTensor}
import com.intel.webscaleml.nn.th.TH
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import scala.sys.process._


class ArrayTensorMathSpec extends FlatSpec with BeforeAndAfter with Matchers{

  before{
    val exitValue = "which th".!
    if(exitValue !== 0){
      cancel("Torch is not installed")
    }
  }

  "matrix + real" should "return correct value" in{
    //user part
    val a = torch.Tensor[Double](200,200).rand()
    val b = 100
    val code = "b = 100\noutcome = a + b"

    val start = System.nanoTime()
    val c = a + b
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a"->a), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== matrix + Real ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime/1e9.toDouble + "\n")

    c should be (luaResult)
  }

  "matrix + matrix" should "return correct value" in{
    //user part
    val a = torch.Tensor[Double](200,200).rand()
    val b = torch.Tensor[Double](200,200).rand()
    val code = "outcome = a + b"

    val start = System.nanoTime()
    val c = a + b
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a"->a, "b"->b), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== matrix + matrix ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime/1e9.toDouble + "\n")

    c should be (luaResult)
  }

  "matrix - real" should "return correct value" in{
    //user part
    val a = torch.Tensor[Double](200,200).rand()
    val b = 100
    val code = "b = 100\noutcome = a - b"

    val start = System.nanoTime()
    val c = a - b
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a"->a), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[DenseTensor[Double]]

    println("\n======== matrix - Real ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime/1e9.toDouble + "\n")

    c should be (luaResult)
  }

  "matrix - matrix" should "return correct value" in{
    //user part
    val a = torch.Tensor[Double](200,200).rand()
    val b = torch.Tensor[Double](200,200).rand()
    val code = "outcome = a - b"

    val start = System.nanoTime()
    val c = a - b
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a"->a, "b"->b), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== matrix - matrix ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime/1e9.toDouble + "\n")

    c should be (luaResult)
  }

  "negative matrix" should "return correct value" in{
    //user part
    val a = torch.Tensor[Double](200,200).rand()
    val code = "outcome = -a"

    val start = System.nanoTime()
    val c = -a
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a"->a), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== negative matrix ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime/1e9.toDouble + "\n")

    c should be (luaResult)
  }

  "matrix / real" should "return correct value" in{
    //user part
    val a = torch.Tensor[Double](200,200).rand()
    val b = 100
    val code = "b = 100\noutcome = a / b"

    val start = System.nanoTime()
    val c = a / b
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a"->a), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== matrix / Real ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime/1e9.toDouble + "\n")

    c should be (luaResult)
  }

  "matrix ./ matrix" should "return correct value" in{
    //user part
    val a = torch.Tensor[Double](200,200).rand()
    val b = torch.Tensor[Double](200,200).rand()
    val code = "outcome = a:cdiv(b)"

    val start = System.nanoTime()
    val c = a / b
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a"->a, "b"->b), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== matrix ./ matrix ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime/1e9.toDouble + "\n")

    c should be (luaResult)
  }

  "matrix * real" should "return correct value" in{
    //user part
    val a = torch.Tensor[Double](200,200).rand()
    val b = 100
    val code = "b = 100\noutcome = a * b"

    val start = System.nanoTime()
    val c = a * b
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a"->a), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== matrix * real ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime/1e9.toDouble + "\n")

    c should be (luaResult)
  }


  "matrix * matrix" should "return correct value" in{
    //user part
    val a = torch.Tensor[Double](200,200).rand()
    val b = torch.Tensor[Double](200,200).rand()
    val code = "outcome = a * b"

    val start = System.nanoTime()
    val c = a * b
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a"->a,"b"->b), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== matrix * matrix ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime/1e9.toDouble + "\n")

    c should be (luaResult)
  }

  "matrix sumAll" should "return correct value" in{
    //user part
    val a = torch.Tensor[Double](200,200).rand()
    val code = "outcome = a:sum()"

    val start = System.nanoTime()
    val c = a.sum()
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a"->a), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Double]

    println("\n======== matrix sumAll ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime/1e9.toDouble + "\n")

    c should be (luaResult)
  }

  "matrix sumDimension" should "return correct value" in{
    //user part
    val a = torch.Tensor[Double](200,200).rand()
    val code = "outcome = a:sum(1)"

    val start = System.nanoTime()
    val c = a.sum(1)
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a"->a), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== matrix sumDimension ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime/1e9.toDouble + "\n")

    c should be (luaResult)
  }

  "matrix max" should "return correct value" in{
    //user part
    val a = torch.Tensor[Double](200,200).rand()
    val code = "outcome = a:max()"

    val start = System.nanoTime()
    val c = a.max()
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a"->a), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Double]

    println("\n======== matrix max ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime/1e9.toDouble + "\n")

    c should be (luaResult)
  }

  "matrix conv2" should "return correct value" in{
    //user part
    val a = torch.Tensor[Double](200,200).rand()
    val b = torch.Tensor[Double](200,200).rand()
    val code = "outcome = torch.conv2(a, b, 'V')"

    val start = System.nanoTime()
    val c = a.conv2(b, 'V')
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a"->a, "b"->b), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== matrix conv2 ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime/1e9.toDouble + "\n")

    c should be (luaResult)
  }

  "matrix xcorr2" should "return correct value" in{
    //user part
    val a = torch.Tensor[Double](200,200).rand()
    val b = torch.Tensor[Double](200,200).rand()
    val code = "outcome = torch.xcorr2(a, b, 'V')"

    val start = System.nanoTime()
    val c = a.xcorr2(b, 'V')
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a"->a, "b"->b), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== matrix xcorr2 ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime/1e9.toDouble + "\n")

    c should be (luaResult)
  }

  "matrix add" should "return correct value" in{
    //user part
    val a = torch.Tensor[Double](200,200).rand()
    val b = torch.Tensor[Double](200,200).rand()
    val m = 20
    val code = "m = 20\noutcome = a:add(m, b)"

    val (luaTime, torchResult) = TH.run(code, Map("a"->a, "b"->b), Array("outcome"))
    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    val start = System.nanoTime()
    val c = a.add(m, b)
    val end = System.nanoTime()
    val scalaTime = end - start

    println("\n======== matrix add ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime/1e9.toDouble + "\n")

    c should be (luaResult)
  }

  "matrix cmul" should "return correct value" in{
    //user part
    val a = torch.Tensor[Double](200,200).rand()
    val b = torch.Tensor[Double](200,200).rand()
    val code = "outcome = a:cmul(b)"

    val (luaTime, torchResult) = TH.run(code, Map("a"->a, "b"->b), Array("outcome"))
    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    val start = System.nanoTime()
    val c = a.cmul(b)
    val end = System.nanoTime()
    val scalaTime = end - start

    println("\n======== matrix cmul ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime/1e9.toDouble + "\n")

    c should be (luaResult)
  }

  "matrix mul" should "return correct value" in{
    //user part
    val a = torch.Tensor[Double](200,200).rand()
    val b = torch.Tensor[Double](200,200).rand()
    val m = 20
    val code = "m = 20\noutcome = a:mul(b, m)"

    val start = System.nanoTime()
    val c = a.mul(b, m)
    val end = System.nanoTime()
    val scalaTime = end - start

    val (luaTime, torchResult) = TH.run(code, Map("a"->a, "b"->b), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== matrix mul ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime/1e9.toDouble + "\n")

    c should be (luaResult)
  }

  "matrix div" should "return correct value" in{
    //user part
    val a = torch.Tensor[Double](200,200).rand()
    val b = 20
    val code = "b = 20\noutcome = a:div(b)"

    val (luaTime, torchResult) = TH.run(code, Map("a"->a), Array("outcome"))
    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    val start = System.nanoTime()
    val c = a.div(b)
    val end = System.nanoTime()
    val scalaTime = end - start


    println("\n======== matrix div ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime/1e9.toDouble + "\n")

    c should be (luaResult)
  }

  "addmm" should "return correct value" in {

    val a = torch.Tensor[Double](200,200).rand()
    val b = torch.Tensor[Double](200,200).rand()

    val start = System.nanoTime()
    val c = torch.Tensor[Double]().resize(Array(a.size(1), b.size(2))).addmm(a, b)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "outcome = torch.Tensor(a:size(1), b:size(2)):fill(0)\noutcome:addmm(a, b)"

    val (luaTime, torchResult) = TH.run(code, Map("a"->a, "b"->b), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== addmm ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime/1e9.toDouble + "\n")

    c.map(luaResult.asInstanceOf[DenseTensor[Double]], (tmp1, tmp2) => {
      tmp1 should be (tmp2 +- 1e-6)
      tmp1
    })
  }

    //TODO: qiuxin, may use it in the future
//  "addmm ()" should "return correct value" in {
//
//    val weight = torch.load("/tmp/w.t7").asInstanceOf[Tensor[Double]]
//
//    val output = torch.load("/tmp/o.t7").asInstanceOf[Tensor[Double]]
//    val input = torch.load("/tmp/f.t7").asInstanceOf[Tensor[Double]]
//
//    val code = "output:addmm(1, output, 1, weight, input)"
//
//    val th = new TH
//    val (luaTime, torchResult) = th.run(code, Map("input"->input, "weight"->weight, "output"->output), Array("output"))
//
//    val start = System.nanoTime()
//    output.addmm(1, output, 1, weight, input)
//    val end = System.nanoTime()
//    val scalaTime = end - start
//
//    val luaResult = torchResult("output").asInstanceOf[Tensor[Double]]
//
//    println("\n======== addmm ========")
//    println("luaTime : " + luaTime)
//    println("scalaTime : " + scalaTime/1e9.toDouble + "\n")
//
//    luaResult should be (output)
//
//  }

  "addr" should "return correct value" in {

    val a = torch.Tensor[Double](200).rand()
    val b = torch.Tensor[Double](200).rand()

    val start = System.nanoTime()
    val c = torch.Tensor[Double]().resize(Array(a.size(1), b.size(1))).addr(a, b)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "outcome = torch.Tensor(a:size(1), b:size(1)):fill(0)\noutcome:addr(a, b)"

    val (luaTime, torchResult) = TH.run(code, Map("a"->a, "b"->b), Array("outcome"))

    val luaResult = torchResult("outcome").asInstanceOf[Tensor[Double]]

    println("\n======== addr ========")
    println("luaTime : " + luaTime)
    println("scalaTime : " + scalaTime/1e9.toDouble + "\n")

    c.map(luaResult.asInstanceOf[DenseTensor[Double]], (tmp1, tmp2) => {
      tmp1 should be (tmp2 +- 1e-6)
      tmp1
    })
  }

}


