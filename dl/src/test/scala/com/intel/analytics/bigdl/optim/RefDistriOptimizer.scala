package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.dataset.{DataSet => DataSource}
import com.intel.analytics.bigdl.nn.{Criterion, Module}
import com.intel.analytics.bigdl.ps.FP16Parameter
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Table, Activities}
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
 * The class is used as a reference optimizer in distribute optimizer unit test
 */
class RefDistriOptimizer[T: ClassTag](
  model: Module[Activities, Activities, T],
  dataset: DataSource[RDD[(Tensor[T], Tensor[T])]],
  criterion: Criterion[Activities, T])(implicit ev : TensorNumeric[T])
  extends Optimizer[T, RDD[(Tensor[T], Tensor[T])], RDD[(Tensor[T], Tensor[T])]](
    model, dataset, criterion
  ){

  override def optimize(): Module[Activities, Activities, T] = {
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
    model: Module[Activities, Activities, T],
    dataset: DataSource[RDD[(Tensor[T], Tensor[T])]],
    criterion: Criterion[Activities, T],
    optimMethod: OptimMethod[T],
    state: Table,
    endWhen: Trigger,
    ev: TensorNumeric[T]
  ): Module[Activities, Activities, T] = {
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
        val fp16W = new FP16Parameter[T](localW)
        fp16W.copyTo(localW)
        val (input, target) = iter.next()
        val output = model.forward(input).asInstanceOf[Tensor[T]]
        val loss = criterion.forward(output, target)
        model.backward(input, criterion.backward(output, target))
        fp16W.copyFrom(localG)
        fp16W.copyTo(localG)
        Iterator.single(loss, localG, input.size(1))
      }).reduce((l, r) => {
        (ev.plus(l._1, r._1), {
          l._2.add(r._2)
          val fp16W = new FP16Parameter[T](l._2)
          fp16W.copyTo(l._2)
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
