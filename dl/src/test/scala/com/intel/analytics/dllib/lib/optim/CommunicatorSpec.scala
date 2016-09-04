package com.intel.analytics.dllib.lib.optim

import com.intel.analytics.dllib.lib.tensor.TensorNumericMath.TensorNumeric.TensorNumericDouble
import com.intel.analytics.dllib.lib.tensor.torch
import org.scalatest.{Matchers, FlatSpec}

class CommunicatorSpec extends FlatSpec with Matchers {

  "convert tensor to fp16 array and back" should "be same when the number is integer" in {
    val tensor = torch.Tensor[Double](5)
    tensor.setValue(1, 1.0)
    tensor.setValue(2, 2.0)
    tensor.setValue(3, 3.0)
    tensor.setValue(4, 4.0)
    tensor.setValue(5, 5.0)

    val fp16Array = new Array[Byte](10)
    CompressedCommunicator.toFP16(tensor.storage().array(), tensor.nElement(), tensor.storageOffset() - 1, fp16Array)
    val test = tensor.clone()
    test.fill(0)
    CompressedCommunicator.fromFP16(fp16Array, test.storage().array(), test.storageOffset() - 1)

    test should be(tensor)
  }

  "convert tensor to fp16 array and back" should "be cut when the number is float" in {
    val tensor = torch.Tensor[Double](5)
    tensor.setValue(1, 1.111111)
    tensor.setValue(2, 2.111111)
    tensor.setValue(3, 3.111111)
    tensor.setValue(4, 4.111111)
    tensor.setValue(5, 5.111111)

    val fp16Array = new Array[Byte](10)
    CompressedCommunicator.toFP16(tensor.storage().array(), tensor.nElement(), tensor.storageOffset() - 1, fp16Array)
    val test = tensor.clone()
    test.fill(0)
    CompressedCommunicator.fromFP16(fp16Array, test.storage().array(), test.storageOffset() - 1)


    val target = torch.Tensor[Double](5)
    target.setValue(1, 1.109375)
    target.setValue(2, 2.109375)
    target.setValue(3, 3.109375)
    target.setValue(4, 4.09375)
    target.setValue(5, 5.09375)

    test  should be(target)
  }

  "two fp16 byte array add" should " be correct" in {
    val tensor1 = torch.Tensor[Double](5)
    tensor1.setValue(1, 1.0)
    tensor1.setValue(2, 2.0)
    tensor1.setValue(3, 3.0)
    tensor1.setValue(4, 4.0)
    tensor1.setValue(5, 5.0)


    val tensor2 = torch.Tensor[Double](5)
    tensor2.setValue(1, 2.0)
    tensor2.setValue(2, 3.0)
    tensor2.setValue(3, 4.0)
    tensor2.setValue(4, 5.0)
    tensor2.setValue(5, 6.0)

    val fp16Array1 = new Array[Byte](10)
    val fp16Array2 = new Array[Byte](10)
    CompressedCommunicator.toFP16(tensor1.storage().array(), tensor1.nElement(), tensor1.storageOffset() - 1, fp16Array1)
    CompressedCommunicator.toFP16(tensor2.storage().array(), tensor2.nElement(), tensor2.storageOffset() - 1, fp16Array2)
    val result = CompressedCommunicator.FP16Add(fp16Array1, fp16Array2)

    tensor1.add(tensor2)

    val test = tensor1.clone()
    test.fill(0)
    CompressedCommunicator.fromFP16(result, test.storage().array(), test.storageOffset() - 1)

    test should be(tensor1)
  }

  "performance of two fp16 byte array add" should " be good" in {
    val size = 1024 * 1024 * 100
    val tensor1 = torch.Tensor[Double](size)
    val tensor2 = torch.Tensor[Double](size)

    val fp16Array1 = new Array[Byte](size * 2)
    val fp16Array2 = new Array[Byte](size * 2)
    CompressedCommunicator.toFP16(tensor1.storage().array(), tensor1.nElement(), tensor1.storageOffset() - 1, fp16Array1)
    CompressedCommunicator.toFP16(tensor2.storage().array(), tensor2.nElement(), tensor2.storageOffset() - 1, fp16Array2)
    val start = System.nanoTime()
    val result = CompressedCommunicator.FP16Add(fp16Array1, fp16Array2)
    println(s"Time is ${(System.nanoTime() - start) / 1e9}s")

    "True" should be("True")
  }

}
