package com.intel.analytics.zoo.feature.common

import com.intel.analytics.bigdl.dataset.Transformer

/**
 * Convert a BigDL Transformer to a Preprocessing
 */
class BigDLAdapter[A, B] (bigDLTransformer: Transformer[A, B]) extends Preprocessing[A, B] {
  override def apply(prev: Iterator[A]): Iterator[B] = {
    bigDLTransformer(prev)
  }
}

object BigDLAdapter {
  def apply[A, B](bigDLTransformer: Transformer[A, B]): BigDLAdapter[A, B] =
    new BigDLAdapter(bigDLTransformer)
}
