package com.intel.analytics.dllib.lib.nn

import org.scalatest.{Matchers, FlatSpec}

class ConcatSpec extends FlatSpec with Matchers {

  "toString" should "return good value" in {
    val seq1 = new Sequential[Double]
    seq1.add(new Linear(10, 15))
    seq1.add(new Sigmoid)

    val seq2 = new Sequential[Double]
    seq2.add(new Linear(10, 15))
    seq2.add(new Tanh)

    val concat = new Concat[Double](2)
    concat.add(seq1)
    concat.add(seq2)

    println(concat)

  }

}
