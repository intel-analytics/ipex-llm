package com.intel.analytics.bigdl.ppml.python

import com.intel.analytics.bigdl.ppml.PPMLContext
import org.scalatest.FunSuite

class PPMLContextWrapperTest extends FunSuite {
  val ppmlContextWrapper: PPMLContextWrapper[Float] = PPMLContextWrapper.ofFloat

  var sc: PPMLContext = null

  test("init with app name") {
    sc = ppmlContextWrapper.createPPMLContext("test")
  }


}
