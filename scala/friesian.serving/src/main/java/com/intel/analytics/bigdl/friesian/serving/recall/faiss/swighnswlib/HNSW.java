/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package com.intel.analytics.bigdl.friesian.serving.recall.faiss.swighnswlib;

public class HNSW {
  private transient long swigCPtr;
  protected transient boolean swigCMemOwn;

  protected HNSW(long cPtr, boolean cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = cPtr;
  }

  protected static long getCPtr(HNSW obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        swigfaissJNI.delete_HNSW(swigCPtr);
      }
      swigCPtr = 0;
    }
  }

  static public class MinimaxHeap {
    private transient long swigCPtr;
    protected transient boolean swigCMemOwn;
  
    protected MinimaxHeap(long cPtr, boolean cMemoryOwn) {
      swigCMemOwn = cMemoryOwn;
      swigCPtr = cPtr;
    }
  
    protected static long getCPtr(MinimaxHeap obj) {
      return (obj == null) ? 0 : obj.swigCPtr;
    }
  
    protected void finalize() {
      delete();
    }
  
    public synchronized void delete() {
      if (swigCPtr != 0) {
        if (swigCMemOwn) {
          swigCMemOwn = false;
          swigfaissJNI.delete_HNSW_MinimaxHeap(swigCPtr);
        }
        swigCPtr = 0;
      }
    }
  
    public void setN(int value) {
      swigfaissJNI.HNSW_MinimaxHeap_n_set(swigCPtr, this, value);
    }
  
    public int getN() {
      return swigfaissJNI.HNSW_MinimaxHeap_n_get(swigCPtr, this);
    }
  
    public void setK(int value) {
      swigfaissJNI.HNSW_MinimaxHeap_k_set(swigCPtr, this, value);
    }
  
    public int getK() {
      return swigfaissJNI.HNSW_MinimaxHeap_k_get(swigCPtr, this);
    }
  
    public void setNvalid(int value) {
      swigfaissJNI.HNSW_MinimaxHeap_nvalid_set(swigCPtr, this, value);
    }
  
    public int getNvalid() {
      return swigfaissJNI.HNSW_MinimaxHeap_nvalid_get(swigCPtr, this);
    }
  
    public void setIds(IntVector value) {
      swigfaissJNI.HNSW_MinimaxHeap_ids_set(swigCPtr, this, IntVector.getCPtr(value), value);
    }
  
    public IntVector getIds() {
      long cPtr = swigfaissJNI.HNSW_MinimaxHeap_ids_get(swigCPtr, this);
      return (cPtr == 0) ? null : new IntVector(cPtr, false);
    }
  
    public void setDis(FloatVector value) {
      swigfaissJNI.HNSW_MinimaxHeap_dis_set(swigCPtr, this, FloatVector.getCPtr(value), value);
    }
  
    public FloatVector getDis() {
      long cPtr = swigfaissJNI.HNSW_MinimaxHeap_dis_get(swigCPtr, this);
      return (cPtr == 0) ? null : new FloatVector(cPtr, false);
    }
  
    public MinimaxHeap(int n) {
      this(swigfaissJNI.new_HNSW_MinimaxHeap(n), true);
    }
  
    public void push(int i, float v) {
      swigfaissJNI.HNSW_MinimaxHeap_push(swigCPtr, this, i, v);
    }
  
    public float max() {
      return swigfaissJNI.HNSW_MinimaxHeap_max(swigCPtr, this);
    }
  
    public int size() {
      return swigfaissJNI.HNSW_MinimaxHeap_size(swigCPtr, this);
    }
  
    public void clear() {
      swigfaissJNI.HNSW_MinimaxHeap_clear(swigCPtr, this);
    }
  
    public int pop_min(SWIGTYPE_p_float vmin_out) {
      return swigfaissJNI.HNSW_MinimaxHeap_pop_min__SWIG_0(swigCPtr, this, SWIGTYPE_p_float.getCPtr(vmin_out));
    }
  
    public int pop_min() {
      return swigfaissJNI.HNSW_MinimaxHeap_pop_min__SWIG_1(swigCPtr, this);
    }
  
    public int count_below(float thresh) {
      return swigfaissJNI.HNSW_MinimaxHeap_count_below(swigCPtr, this, thresh);
    }
  
  }

  static public class NodeDistCloser {
    private transient long swigCPtr;
    protected transient boolean swigCMemOwn;
  
    protected NodeDistCloser(long cPtr, boolean cMemoryOwn) {
      swigCMemOwn = cMemoryOwn;
      swigCPtr = cPtr;
    }
  
    protected static long getCPtr(NodeDistCloser obj) {
      return (obj == null) ? 0 : obj.swigCPtr;
    }
  
    protected void finalize() {
      delete();
    }
  
    public synchronized void delete() {
      if (swigCPtr != 0) {
        if (swigCMemOwn) {
          swigCMemOwn = false;
          swigfaissJNI.delete_HNSW_NodeDistCloser(swigCPtr);
        }
        swigCPtr = 0;
      }
    }
  
    public void setD(float value) {
      swigfaissJNI.HNSW_NodeDistCloser_d_set(swigCPtr, this, value);
    }
  
    public float getD() {
      return swigfaissJNI.HNSW_NodeDistCloser_d_get(swigCPtr, this);
    }
  
    public void setId(int value) {
      swigfaissJNI.HNSW_NodeDistCloser_id_set(swigCPtr, this, value);
    }
  
    public int getId() {
      return swigfaissJNI.HNSW_NodeDistCloser_id_get(swigCPtr, this);
    }
  
    public NodeDistCloser(float d, int id) {
      this(swigfaissJNI.new_HNSW_NodeDistCloser(d, id), true);
    }
  
  }

  static public class NodeDistFarther {
    private transient long swigCPtr;
    protected transient boolean swigCMemOwn;
  
    protected NodeDistFarther(long cPtr, boolean cMemoryOwn) {
      swigCMemOwn = cMemoryOwn;
      swigCPtr = cPtr;
    }
  
    protected static long getCPtr(NodeDistFarther obj) {
      return (obj == null) ? 0 : obj.swigCPtr;
    }
  
    protected void finalize() {
      delete();
    }
  
    public synchronized void delete() {
      if (swigCPtr != 0) {
        if (swigCMemOwn) {
          swigCMemOwn = false;
          swigfaissJNI.delete_HNSW_NodeDistFarther(swigCPtr);
        }
        swigCPtr = 0;
      }
    }
  
    public void setD(float value) {
      swigfaissJNI.HNSW_NodeDistFarther_d_set(swigCPtr, this, value);
    }
  
    public float getD() {
      return swigfaissJNI.HNSW_NodeDistFarther_d_get(swigCPtr, this);
    }
  
    public void setId(int value) {
      swigfaissJNI.HNSW_NodeDistFarther_id_set(swigCPtr, this, value);
    }
  
    public int getId() {
      return swigfaissJNI.HNSW_NodeDistFarther_id_get(swigCPtr, this);
    }
  
    public NodeDistFarther(float d, int id) {
      this(swigfaissJNI.new_HNSW_NodeDistFarther(d, id), true);
    }
  
  }

  public void setAssign_probas(DoubleVector value) {
    swigfaissJNI.HNSW_assign_probas_set(swigCPtr, this, DoubleVector.getCPtr(value), value);
  }

  public DoubleVector getAssign_probas() {
    long cPtr = swigfaissJNI.HNSW_assign_probas_get(swigCPtr, this);
    return (cPtr == 0) ? null : new DoubleVector(cPtr, false);
  }

  public void setCum_nneighbor_per_level(IntVector value) {
    swigfaissJNI.HNSW_cum_nneighbor_per_level_set(swigCPtr, this, IntVector.getCPtr(value), value);
  }

  public IntVector getCum_nneighbor_per_level() {
    long cPtr = swigfaissJNI.HNSW_cum_nneighbor_per_level_get(swigCPtr, this);
    return (cPtr == 0) ? null : new IntVector(cPtr, false);
  }

  public void setLevels(IntVector value) {
    swigfaissJNI.HNSW_levels_set(swigCPtr, this, IntVector.getCPtr(value), value);
  }

  public IntVector getLevels() {
    long cPtr = swigfaissJNI.HNSW_levels_get(swigCPtr, this);
    return (cPtr == 0) ? null : new IntVector(cPtr, false);
  }

  public void setOffsets(Uint64Vector value) {
    swigfaissJNI.HNSW_offsets_set(swigCPtr, this, Uint64Vector.getCPtr(value), value);
  }

  public Uint64Vector getOffsets() {
    long cPtr = swigfaissJNI.HNSW_offsets_get(swigCPtr, this);
    return (cPtr == 0) ? null : new Uint64Vector(cPtr, false);
  }

  public void setNeighbors(IntVector value) {
    swigfaissJNI.HNSW_neighbors_set(swigCPtr, this, IntVector.getCPtr(value), value);
  }

  public IntVector getNeighbors() {
    long cPtr = swigfaissJNI.HNSW_neighbors_get(swigCPtr, this);
    return (cPtr == 0) ? null : new IntVector(cPtr, false);
  }

  public void setEntry_point(int value) {
    swigfaissJNI.HNSW_entry_point_set(swigCPtr, this, value);
  }

  public int getEntry_point() {
    return swigfaissJNI.HNSW_entry_point_get(swigCPtr, this);
  }

  public void setRng(RandomGenerator value) {
    swigfaissJNI.HNSW_rng_set(swigCPtr, this, RandomGenerator.getCPtr(value), value);
  }

  public RandomGenerator getRng() {
    long cPtr = swigfaissJNI.HNSW_rng_get(swigCPtr, this);
    return (cPtr == 0) ? null : new RandomGenerator(cPtr, false);
  }

  public void setMax_level(int value) {
    swigfaissJNI.HNSW_max_level_set(swigCPtr, this, value);
  }

  public int getMax_level() {
    return swigfaissJNI.HNSW_max_level_get(swigCPtr, this);
  }

  public void setEfConstruction(int value) {
    swigfaissJNI.HNSW_efConstruction_set(swigCPtr, this, value);
  }

  public int getEfConstruction() {
    return swigfaissJNI.HNSW_efConstruction_get(swigCPtr, this);
  }

  public void setEfSearch(int value) {
    swigfaissJNI.HNSW_efSearch_set(swigCPtr, this, value);
  }

  public int getEfSearch() {
    return swigfaissJNI.HNSW_efSearch_get(swigCPtr, this);
  }

  public void setCheck_relative_distance(boolean value) {
    swigfaissJNI.HNSW_check_relative_distance_set(swigCPtr, this, value);
  }

  public boolean getCheck_relative_distance() {
    return swigfaissJNI.HNSW_check_relative_distance_get(swigCPtr, this);
  }

  public void setUpper_beam(int value) {
    swigfaissJNI.HNSW_upper_beam_set(swigCPtr, this, value);
  }

  public int getUpper_beam() {
    return swigfaissJNI.HNSW_upper_beam_get(swigCPtr, this);
  }

  public void setSearch_bounded_queue(boolean value) {
    swigfaissJNI.HNSW_search_bounded_queue_set(swigCPtr, this, value);
  }

  public boolean getSearch_bounded_queue() {
    return swigfaissJNI.HNSW_search_bounded_queue_get(swigCPtr, this);
  }

  public void set_default_probas(int M, float levelMult) {
    swigfaissJNI.HNSW_set_default_probas(swigCPtr, this, M, levelMult);
  }

  public void set_nb_neighbors(int level_no, int n) {
    swigfaissJNI.HNSW_set_nb_neighbors(swigCPtr, this, level_no, n);
  }

  public int nb_neighbors(int layer_no) {
    return swigfaissJNI.HNSW_nb_neighbors(swigCPtr, this, layer_no);
  }

  public int cum_nb_neighbors(int layer_no) {
    return swigfaissJNI.HNSW_cum_nb_neighbors(swigCPtr, this, layer_no);
  }

  public void neighbor_range(int no, int layer_no, SWIGTYPE_p_unsigned_long begin, SWIGTYPE_p_unsigned_long end) {
    swigfaissJNI.HNSW_neighbor_range(swigCPtr, this, no, layer_no, SWIGTYPE_p_unsigned_long.getCPtr(begin), SWIGTYPE_p_unsigned_long.getCPtr(end));
  }

  public HNSW(int M) {
    this(swigfaissJNI.new_HNSW__SWIG_0(M), true);
  }

  public HNSW() {
    this(swigfaissJNI.new_HNSW__SWIG_1(), true);
  }

  public int random_level() {
    return swigfaissJNI.HNSW_random_level(swigCPtr, this);
  }

  public void fill_with_random_links(long n) {
    swigfaissJNI.HNSW_fill_with_random_links(swigCPtr, this, n);
  }

  public void add_links_starting_from(DistanceComputer ptdis, int pt_id, int nearest, float d_nearest, int level, SWIGTYPE_p_omp_lock_t locks, VisitedTable vt) {
    swigfaissJNI.HNSW_add_links_starting_from(swigCPtr, this, DistanceComputer.getCPtr(ptdis), ptdis, pt_id, nearest, d_nearest, level, SWIGTYPE_p_omp_lock_t.getCPtr(locks), VisitedTable.getCPtr(vt), vt);
  }

  public void add_with_locks(DistanceComputer ptdis, int pt_level, int pt_id, SWIGTYPE_p_std__vectorT_omp_lock_t_t locks, VisitedTable vt) {
    swigfaissJNI.HNSW_add_with_locks(swigCPtr, this, DistanceComputer.getCPtr(ptdis), ptdis, pt_level, pt_id, SWIGTYPE_p_std__vectorT_omp_lock_t_t.getCPtr(locks), VisitedTable.getCPtr(vt), vt);
  }

  public int search_from_candidates(DistanceComputer qdis, int k, SWIGTYPE_p_long I, SWIGTYPE_p_float D, HNSW.MinimaxHeap candidates, VisitedTable vt, int level, int nres_in) {
    return swigfaissJNI.HNSW_search_from_candidates__SWIG_0(swigCPtr, this, DistanceComputer.getCPtr(qdis), qdis, k, SWIGTYPE_p_long.getCPtr(I), SWIGTYPE_p_float.getCPtr(D), HNSW.MinimaxHeap.getCPtr(candidates), candidates, VisitedTable.getCPtr(vt), vt, level, nres_in);
  }

  public int search_from_candidates(DistanceComputer qdis, int k, SWIGTYPE_p_long I, SWIGTYPE_p_float D, HNSW.MinimaxHeap candidates, VisitedTable vt, int level) {
    return swigfaissJNI.HNSW_search_from_candidates__SWIG_1(swigCPtr, this, DistanceComputer.getCPtr(qdis), qdis, k, SWIGTYPE_p_long.getCPtr(I), SWIGTYPE_p_float.getCPtr(D), HNSW.MinimaxHeap.getCPtr(candidates), candidates, VisitedTable.getCPtr(vt), vt, level);
  }

  public SWIGTYPE_p_std__priority_queueT_std__pairT_float_int_t_t search_from_candidate_unbounded(SWIGTYPE_p_std__pairT_float_int_t node, DistanceComputer qdis, int ef, VisitedTable vt) {
    return new SWIGTYPE_p_std__priority_queueT_std__pairT_float_int_t_t(swigfaissJNI.HNSW_search_from_candidate_unbounded(swigCPtr, this, SWIGTYPE_p_std__pairT_float_int_t.getCPtr(node), DistanceComputer.getCPtr(qdis), qdis, ef, VisitedTable.getCPtr(vt), vt), true);
  }

  public void search(DistanceComputer qdis, int k, SWIGTYPE_p_long I, SWIGTYPE_p_float D, VisitedTable vt) {
    swigfaissJNI.HNSW_search(swigCPtr, this, DistanceComputer.getCPtr(qdis), qdis, k, SWIGTYPE_p_long.getCPtr(I), SWIGTYPE_p_float.getCPtr(D), VisitedTable.getCPtr(vt), vt);
  }

  public void reset() {
    swigfaissJNI.HNSW_reset(swigCPtr, this);
  }

  public void clear_neighbor_tables(int level) {
    swigfaissJNI.HNSW_clear_neighbor_tables(swigCPtr, this, level);
  }

  public void print_neighbor_stats(int level) {
    swigfaissJNI.HNSW_print_neighbor_stats(swigCPtr, this, level);
  }

  public int prepare_level_tab(long n, boolean preset_levels) {
    return swigfaissJNI.HNSW_prepare_level_tab__SWIG_0(swigCPtr, this, n, preset_levels);
  }

  public int prepare_level_tab(long n) {
    return swigfaissJNI.HNSW_prepare_level_tab__SWIG_1(swigCPtr, this, n);
  }

  public static void shrink_neighbor_list(DistanceComputer qdis, SWIGTYPE_p_std__priority_queueT_faiss__HNSW__NodeDistFarther_t input, SWIGTYPE_p_std__vectorT_faiss__HNSW__NodeDistFarther_t output, int max_size) {
    swigfaissJNI.HNSW_shrink_neighbor_list(DistanceComputer.getCPtr(qdis), qdis, SWIGTYPE_p_std__priority_queueT_faiss__HNSW__NodeDistFarther_t.getCPtr(input), SWIGTYPE_p_std__vectorT_faiss__HNSW__NodeDistFarther_t.getCPtr(output), max_size);
  }

}
