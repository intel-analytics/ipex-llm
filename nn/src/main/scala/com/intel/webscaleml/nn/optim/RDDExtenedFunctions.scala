package org.apache.spark

import org.apache.spark.rdd.{CoalescedRDD, RDD}
import org.apache.spark.util.Utils
import org.apache.spark.util.collection.OpenHashSet
import scala.reflect.{classTag, ClassTag}

class RDDExtenedFunctions[T: ClassTag](self: RDD[T]) extends Serializable {
    def reductionTreeAggregate[U: ClassTag](zeroValue: U)(
      seqOp: (U, T) => U,
      combOp: (U, U) => U,
      depth: Int = 2,
      balanceSlack: Double = 1.0): U = self.withScope {
      println("Enter reductionTreeAggregate")

      if (self.partitions.length == 0) {
        Utils.clone(zeroValue, self.context.env.closureSerializer.newInstance())
      } else {
        val cleanSeqOp = self.context.clean(seqOp)
        val cleanCombOp = self.context.clean(combOp)
        val aggregatePartition =
          (it: Iterator[T]) => it.aggregate(zeroValue)(cleanSeqOp, cleanCombOp)

        var prePartiallyAggregated: RDD[U] = null
        var partiallyAggregated = self.mapPartitions(it => Iterator(aggregatePartition(it))).cache()
        partiallyAggregated.count()
        partiallyAggregated.count() //Need count 2 times, or preferredLocs is still hdfs location, still investigate reason.

        val hashset = new OpenHashSet[String]()
        partiallyAggregated.partitions.foreach { p =>
          partiallyAggregated.context.getPreferredLocs(partiallyAggregated, p.index).map(tl => tl.host)
                    .foreach(hashset.add(_))
        }

        val executorNum = hashset.size
        prePartiallyAggregated = partiallyAggregated
        partiallyAggregated = new CoalescedRDD[U](partiallyAggregated, executorNum, balanceSlack)
          .mapPartitions(it => Iterator(it.reduce(cleanCombOp))).cache()
        partiallyAggregated.count()

        var numPartitions = executorNum
        val scale = math.max(math.ceil(math.pow(numPartitions, 1.0 / depth)).toInt, 2)
        while (numPartitions > scale + math.ceil(numPartitions.toDouble / scale)) {
          numPartitions /= scale
          prePartiallyAggregated.unpersist()
          prePartiallyAggregated = partiallyAggregated
          partiallyAggregated = new CoalescedRDD[U](partiallyAggregated, numPartitions, balanceSlack)
            .mapPartitions(it => Iterator(it.reduce(cleanCombOp))).cache()
          partiallyAggregated.count()
        }
        val res = partiallyAggregated.reduce(cleanCombOp)
        prePartiallyAggregated.unpersist()
        partiallyAggregated.unpersist()
        res
      }
    }
}
