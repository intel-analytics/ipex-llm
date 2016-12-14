package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.dataset.{DataSet => DataSource, Batch}
import com.intel.analytics.bigdl.parameters.FP16CompressedTensor
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Table}
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
 * The class is used as a reference optimizer in distribute optimizer unit test
 */
class RefDistriOptimizer[T: ClassTag](
  model: Module[T],
  dataset: DataSource[RDD[Batch[T]]],
  criterion: Criterion[T])(implicit ev : TensorNumeric[T])
  extends Optimizer[T, RDD[Batch[T]], RDD[Batch[T]]](
    model, dataset, criterion
  ){

  override def optimize(): Module[T] = {
    RefDistriOptimizer.optimize(
      model,
      dataset,
      criterion,
      optimMethod,
      state,
      endWhen,
      ev
    )
  }
}

object RefDistriOptimizer {
  def optimize[T : ClassTag](
    model: Module[T],
    dataset: DataSource[RDD[Batch[T]]],
    criterion: Criterion[T],
    optimMethod: OptimMethod[T],
    state: Table,
    endWhen: Trigger,
    ev: TensorNumeric[T]
  ): Module[T] = {
    val (w, g) = model.getParameters()
    var count = 0
    state("epoch") = state.get[Int]("epoch").getOrElse(1)
    state("neval") = state.get[Int]("neval").getOrElse(1)
    val partitionNum = dataset.data().partitions.length
    model.training()
    while(!endWhen(state)) {
      val data = dataset.data()
      val (lossSum, grad, batch) = data.mapPartitions(iter => {
        val (localW, localG) = model.getParameters()
        model.zeroGradParameters()
        val fp16W = new FP16CompressedTensor[T](localW)
        fp16W.deCompress(localW)
        val batch = iter.next()
        val input = batch.data
        val target = batch.labels
        val output = model.forward(input).asInstanceOf[Tensor[T]]
        val loss = criterion.forward(output, target)
        model.backward(input, criterion.backward(output, target))
        fp16W.compress(localG)
        fp16W.deCompress(localG)
        Iterator.single(loss, localG, input.size(1))
      }).reduce((l, r) => {
        (ev.plus(l._1, r._1), {
          l._2.add(r._2)
          val fp16W = new FP16CompressedTensor[T](l._2)
          fp16W.deCompress(l._2)
          l._2
        }, l._3 + r._3)
      })
      val loss = ev.divide(lossSum, ev.fromType(partitionNum))
      val gradients = grad.div(ev.fromType(partitionNum))
      optimMethod.optimize(_ => (loss, gradients), w, state)
      count += batch
      state("neval") = state[Int]("neval") + 1
      println(s"loss is $loss")
      if(count >= dataset.size()) {
        state("epoch") = state[Int]("epoch") + 1
        count = 0
      }
    }

    model
  }
}
