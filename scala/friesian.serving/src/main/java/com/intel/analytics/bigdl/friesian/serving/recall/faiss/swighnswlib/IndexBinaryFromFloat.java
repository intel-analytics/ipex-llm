/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package com.intel.analytics.bigdl.friesian.serving.recall.faiss.swighnswlib;

public class IndexBinaryFromFloat extends IndexBinary {
  private transient long swigCPtr;

  protected IndexBinaryFromFloat(long cPtr, boolean cMemoryOwn) {
    super(swigfaissJNI.IndexBinaryFromFloat_SWIGUpcast(cPtr), cMemoryOwn);
    swigCPtr = cPtr;
  }

  protected static long getCPtr(IndexBinaryFromFloat obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        swigfaissJNI.delete_IndexBinaryFromFloat(swigCPtr);
      }
      swigCPtr = 0;
    }
    super.delete();
  }

  public void setIndex(Index value) {
    swigfaissJNI.IndexBinaryFromFloat_index_set(swigCPtr, this, Index.getCPtr(value), value);
  }

  public Index getIndex() {
    long cPtr = swigfaissJNI.IndexBinaryFromFloat_index_get(swigCPtr, this);
    return (cPtr == 0) ? null : new Index(cPtr, false);
  }

  public void setOwn_fields(boolean value) {
    swigfaissJNI.IndexBinaryFromFloat_own_fields_set(swigCPtr, this, value);
  }

  public boolean getOwn_fields() {
    return swigfaissJNI.IndexBinaryFromFloat_own_fields_get(swigCPtr, this);
  }

  public IndexBinaryFromFloat() {
    this(swigfaissJNI.new_IndexBinaryFromFloat__SWIG_0(), true);
  }

  public IndexBinaryFromFloat(Index index) {
    this(swigfaissJNI.new_IndexBinaryFromFloat__SWIG_1(Index.getCPtr(index), index), true);
  }

  public void add(int n, SWIGTYPE_p_unsigned_char x) {
    swigfaissJNI.IndexBinaryFromFloat_add(swigCPtr, this, n, SWIGTYPE_p_unsigned_char.getCPtr(x));
  }

  public void reset() {
    swigfaissJNI.IndexBinaryFromFloat_reset(swigCPtr, this);
  }

  public void search(int n, SWIGTYPE_p_unsigned_char x, int k, SWIGTYPE_p_int distances, SWIGTYPE_p_long labels) {
    swigfaissJNI.IndexBinaryFromFloat_search(swigCPtr, this, n, SWIGTYPE_p_unsigned_char.getCPtr(x), k, SWIGTYPE_p_int.getCPtr(distances), SWIGTYPE_p_long.getCPtr(labels));
  }

  public void train(int n, SWIGTYPE_p_unsigned_char x) {
    swigfaissJNI.IndexBinaryFromFloat_train(swigCPtr, this, n, SWIGTYPE_p_unsigned_char.getCPtr(x));
  }

}
