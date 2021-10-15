/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package com.intel.analytics.bigdl.friesian.serving.recall.faiss.swighnswlib;

public class Index2Layer extends Index {
  private transient long swigCPtr;

  protected Index2Layer(long cPtr, boolean cMemoryOwn) {
    super(swigfaissJNI.Index2Layer_SWIGUpcast(cPtr), cMemoryOwn);
    swigCPtr = cPtr;
  }

  protected static long getCPtr(Index2Layer obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        swigfaissJNI.delete_Index2Layer(swigCPtr);
      }
      swigCPtr = 0;
    }
    super.delete();
  }

  public void setQ1(Level1Quantizer value) {
    swigfaissJNI.Index2Layer_q1_set(swigCPtr, this, Level1Quantizer.getCPtr(value), value);
  }

  public Level1Quantizer getQ1() {
    long cPtr = swigfaissJNI.Index2Layer_q1_get(swigCPtr, this);
    return (cPtr == 0) ? null : new Level1Quantizer(cPtr, false);
  }

  public void setPq(ProductQuantizer value) {
    swigfaissJNI.Index2Layer_pq_set(swigCPtr, this, ProductQuantizer.getCPtr(value), value);
  }

  public ProductQuantizer getPq() {
    long cPtr = swigfaissJNI.Index2Layer_pq_get(swigCPtr, this);
    return (cPtr == 0) ? null : new ProductQuantizer(cPtr, false);
  }

  public void setCodes(ByteVector value) {
    swigfaissJNI.Index2Layer_codes_set(swigCPtr, this, ByteVector.getCPtr(value), value);
  }

  public ByteVector getCodes() {
    long cPtr = swigfaissJNI.Index2Layer_codes_get(swigCPtr, this);
    return (cPtr == 0) ? null : new ByteVector(cPtr, false);
  }

  public void setCode_size_1(long value) {
    swigfaissJNI.Index2Layer_code_size_1_set(swigCPtr, this, value);
  }

  public long getCode_size_1() {
    return swigfaissJNI.Index2Layer_code_size_1_get(swigCPtr, this);
  }

  public void setCode_size_2(long value) {
    swigfaissJNI.Index2Layer_code_size_2_set(swigCPtr, this, value);
  }

  public long getCode_size_2() {
    return swigfaissJNI.Index2Layer_code_size_2_get(swigCPtr, this);
  }

  public void setCode_size(long value) {
    swigfaissJNI.Index2Layer_code_size_set(swigCPtr, this, value);
  }

  public long getCode_size() {
    return swigfaissJNI.Index2Layer_code_size_get(swigCPtr, this);
  }

  public Index2Layer(Index quantizer, long nlist, int M, int nbit, MetricType metric) {
    this(swigfaissJNI.new_Index2Layer__SWIG_0(getCPtr(quantizer), quantizer, nlist, M, nbit, metric.swigValue()), true);
  }

  public Index2Layer(Index quantizer, long nlist, int M, int nbit) {
    this(swigfaissJNI.new_Index2Layer__SWIG_1(getCPtr(quantizer), quantizer, nlist, M, nbit), true);
  }

  public Index2Layer(Index quantizer, long nlist, int M) {
    this(swigfaissJNI.new_Index2Layer__SWIG_2(getCPtr(quantizer), quantizer, nlist, M), true);
  }

  public Index2Layer() {
    this(swigfaissJNI.new_Index2Layer__SWIG_3(), true);
  }

  public void train(int n, SWIGTYPE_p_float x) {
    swigfaissJNI.Index2Layer_train(swigCPtr, this, n, SWIGTYPE_p_float.getCPtr(x));
  }

  public void add(int n, SWIGTYPE_p_float x) {
    swigfaissJNI.Index2Layer_add(swigCPtr, this, n, SWIGTYPE_p_float.getCPtr(x));
  }

  public void search(int n, SWIGTYPE_p_float x, int k, SWIGTYPE_p_float distances, SWIGTYPE_p_long labels) {
    swigfaissJNI.Index2Layer_search(swigCPtr, this, n, SWIGTYPE_p_float.getCPtr(x), k, SWIGTYPE_p_float.getCPtr(distances), SWIGTYPE_p_long.getCPtr(labels));
  }

  public void reconstruct_n(int i0, int ni, SWIGTYPE_p_float recons) {
    swigfaissJNI.Index2Layer_reconstruct_n(swigCPtr, this, i0, ni, SWIGTYPE_p_float.getCPtr(recons));
  }

  public void reconstruct(int key, SWIGTYPE_p_float recons) {
    swigfaissJNI.Index2Layer_reconstruct(swigCPtr, this, key, SWIGTYPE_p_float.getCPtr(recons));
  }

  public void reset() {
    swigfaissJNI.Index2Layer_reset(swigCPtr, this);
  }

  public DistanceComputer get_distance_computer() {
    long cPtr = swigfaissJNI.Index2Layer_get_distance_computer(swigCPtr, this);
    return (cPtr == 0) ? null : new DistanceComputer(cPtr, true);
  }

  public void transfer_to_IVFPQ(IndexIVFPQ other) {
    swigfaissJNI.Index2Layer_transfer_to_IVFPQ(swigCPtr, this, IndexIVFPQ.getCPtr(other), other);
  }

  public long sa_code_size() {
    return swigfaissJNI.Index2Layer_sa_code_size(swigCPtr, this);
  }

  public void sa_encode(int n, SWIGTYPE_p_float x, SWIGTYPE_p_unsigned_char bytes) {
    swigfaissJNI.Index2Layer_sa_encode(swigCPtr, this, n, SWIGTYPE_p_float.getCPtr(x), SWIGTYPE_p_unsigned_char.getCPtr(bytes));
  }

  public void sa_decode(int n, SWIGTYPE_p_unsigned_char bytes, SWIGTYPE_p_float x) {
    swigfaissJNI.Index2Layer_sa_decode(swigCPtr, this, n, SWIGTYPE_p_unsigned_char.getCPtr(bytes), SWIGTYPE_p_float.getCPtr(x));
  }

}
