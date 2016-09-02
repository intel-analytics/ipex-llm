package com.intel.analytics.dllib.lib.nn

import com.intel.analytics.dllib.lib.tensor.{torch, DenseTensor$}
import org.scalatest.FlatSpec

class ReshapeSpec extends FlatSpec{
  "A Reshape Module " should "generate correct output and grad" in {
    val module = new Reshape[Double](Array(3,2))
    for(batchSize<-1 to 4){
      val input = torch.Tensor[Double](batchSize,1,6)
      input.rand()
      val inputOrg = input.clone()
      val output = module.forward(input)
      val gradOutput = torch.Tensor[Double](batchSize,3,2)
      gradOutput.rand()
      val gradOutputOrg = gradOutput.clone()
      val gradInput = module.backward(input,gradOutput)
      assert(output.nDimension() == 3)
      assert(output.size(1) == batchSize)
      assert(output.size(2) == 3)
      assert(output.size(3) == 2)
      assert(gradInput.isSameSizeAs(input))
      for(i<-1 to batchSize) {
        for(j<-0 to 5) {
          assert(input(Array(i,1,j+1)) == output(Array(i,j/2+1,j%2+1)))
          assert(gradInput(Array(i,1,j+1)) == gradOutput(Array(i,j/2+1,j%2+1)))
        }
      }
      assert(input == inputOrg)
      assert(gradOutput == gradOutputOrg)
    }

    intercept[IllegalArgumentException]{
      module.forward(torch.Tensor[Double](2,2))
    }

    intercept[IllegalArgumentException]{
      module.forward(torch.Tensor[Double](3,2,2))
    }
  }

  "A Reshape Module default batch" should "generate correct output and grad" in {
    val module = new Reshape[Double](Array(3,2))
    val input = torch.Tensor[Double](2,3)
    input.rand()
    val inputOrg = input.clone()
    val output = module.forward(input)
    val gradOutput = torch.Tensor[Double](3,2)
    gradOutput.rand()
    val gradOutputOrg = gradOutput.clone()
    val gradInput = module.backward(input,gradOutput)
    assert(output.nDimension() == 2)
    assert(output.size(1) == 3)
    assert(output.size(2) == 2)
    for(j<-0 to 5) {
      assert(input(Array(j/3+1,j%3+1)) == output(Array(j/2+1,j%2+1)))
      assert(gradInput(Array(j/3+1,j%3+1)) == gradOutput(Array(j/2+1,j%2+1)))
    }
    assert(input == inputOrg)
    assert(gradOutput == gradOutputOrg)
  }

  "A Reshape Module disable batch" should "generate correct output and grad" in {
    val module = new Reshape[Double](Array(3,2),Some(false))
    val input = torch.Tensor[Double](1,2,3)
    input.rand()
    val inputOrg = input.clone()
    val output = module.forward(input)
    val gradOutput = torch.Tensor[Double](3,2)
    gradOutput.rand()
    val gradOutputOrg = gradOutput.clone()
    val gradInput = module.backward(input,gradOutput)
    assert(output.nDimension() == 2)
    assert(output.size(1) == 3)
    assert(output.size(2) == 2)
    for(j<-0 to 5) {
      assert(input(Array(1,j/3+1,j%3+1)) == output(Array(j/2+1,j%2+1)))
      assert(gradInput(Array(1,j/3+1,j%3+1)) == gradOutput(Array(j/2+1,j%2+1)))
    }
    assert(input == inputOrg)
    assert(gradOutput == gradOutputOrg)

    intercept[IllegalArgumentException]{
      module.forward(torch.Tensor[Double](2,3,2))
    }
  }

  "A Reshape Module enable batch" should "generate correct output and grad" in {
    val module = new Reshape[Double](Array(3,2),Some(true))
    for(batchSize<-1 to 4){
      val input = torch.Tensor[Double](batchSize,1,6)
      input.rand()
      val inputOrg = input.clone()
      val output = module.forward(input)
      val gradOutput = torch.Tensor[Double](batchSize,3,2)
      gradOutput.rand()
      val gradOutputOrg = gradOutput.clone()
      val gradInput = module.backward(input,gradOutput)
      assert(output.nDimension() == 3)
      assert(output.size(1) == batchSize)
      assert(output.size(2) == 3)
      assert(output.size(3) == 2)
      assert(gradInput.isSameSizeAs(input))
      for(i<-1 to batchSize) {
        for(j<-0 to 5) {
          assert(input(Array(i,1,j+1)) == output(Array(i,j/2+1,j%2+1)))
          assert(gradInput(Array(i,1,j+1)) == gradOutput(Array(i,j/2+1,j%2+1)))
        }
      }
      assert(input == inputOrg)
      assert(gradOutput == gradOutputOrg)
    }

    intercept[IllegalArgumentException]{
      module.forward(torch.Tensor[Double](3,2))
    }
  }
}
