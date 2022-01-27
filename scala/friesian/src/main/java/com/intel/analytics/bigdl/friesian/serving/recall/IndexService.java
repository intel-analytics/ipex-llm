/*
 * Copyright 2021 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.friesian.serving.recall;

import com.google.common.base.Preconditions;
import com.intel.analytics.bigdl.friesian.serving.recall.faiss.swighnswlib.*;
import com.intel.analytics.bigdl.friesian.serving.recall.faiss.utils.JniFaissInitializer;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.List;

public class IndexService {
    private Index index;
    private static final Logger logger = LogManager.getLogger(IndexService.class.getName());
    private static final int efConstruction = 40;
    private static final int efSearch = 256;

    public IndexService(int dim) {
        Preconditions.checkArgument(JniFaissInitializer.initialized());
        index = swigfaiss.index_factory(dim, "HNSWlibInt16_32",
                MetricType.METRIC_INNER_PRODUCT);
        new ParameterSpace().set_index_parameter(index, "efSearch", efSearch);
        new ParameterSpace().set_index_parameter(index, "efConstruction", efConstruction);
        // 32768 for Int16
        new ParameterSpace().set_index_parameter(index, "scale", 32768);
    }

    public void add(int targetId, floatArray data) {
        longArray la = new longArray(1);
        la.setitem(0, targetId);
        index.add_with_ids(1, data.cast(), la.cast());
    }

    public void addWithIds(float[] data, int[] ids) {
        assert(ids.length == data.length);
        int dataNum = ids.length;
        longArray idsInput = convertIdsToLongArray(ids);
        floatArray dataInput = vectorToFloatArray(data);
        long start = System.nanoTime();
        addWithIds(dataInput, idsInput, dataNum);
        long end = System.nanoTime();
        long time = (end - start);
        logger.info("Add " + dataNum + " items to index time: " + time + " ns");
        logger.info("Current NTotal: " + this.getNTotal());
    }

    public void addWithIds(floatArray data, longArray ids, int dataNum) {
        index.add_with_ids(dataNum, data.cast(), ids.cast());
    }
    public void save(String path) {
        swigfaiss.write_index(this.index, path);
    }

    public void load(String path) {
        logger.info("Loading existing index from " + path + "...");
        this.index = swigfaiss.read_index(path);
        new ParameterSpace().set_index_parameter(this.index, "efSearch", efSearch);
        // 32768 for Int16
        new ParameterSpace().set_index_parameter(this.index, "scale", 32768);
    }

    public int getNTotal() {
        return this.index.getNtotal();
    }

    public boolean isTrained() {
        return this.index.getIs_trained();
    }

    public void train(int dataSize, floatArray xb) {
        logger.info("Start training");
        long start = System.nanoTime();
        this.index.train(dataSize, xb.cast());
        long end = System.nanoTime();
        long time = (end - start);
        logger.info("Training time: " + time + " ns");
    }

    public int[] search(floatArray query, int k) {
        longArray I = new longArray(k);
        floatArray D = new floatArray(k);
        index.search(1, query.cast(), k, D.cast(), I.cast());
//        logger.info(show(I, 1, k));
//        logger.info(show(D, 1, k));
        int[] candidates = new int[k];
        for (int i = 0; i < k; i ++) {
            candidates[i] = I.getitem(i);
        }
        return candidates;
    }

    public static floatArray vectorToFloatArray(float[] vector) {
        int d = vector.length;
        floatArray fa = new floatArray(d);
        for (int j = 0; j < d; j++) {
            fa.setitem(j, vector[j]);
        }
        return fa;
    }

    public static floatArray listOfVectorToFloatArray(List<float[]> vectors) {
        int nb = vectors.size();
        int d = vectors.get(0).length;
        floatArray fa = new floatArray(d * nb);
        for (int i = 0; i < nb; i++) {
            float[] vector = vectors.get(i);
            for (int j = 0; j < d; j++) {
                fa.setitem(d * i + j, vector[j]);
            }
        }
        return fa;
    }

    public static longArray convertIdsToLongArray(List<Integer> ids) {
        int[] idArr = ids.stream().mapToInt(i -> i).toArray();
        return convertIdsToLongArray(idArr);
    }

    public static longArray convertIdsToLongArray(int[] ids) {
        int num = ids.length;
        longArray la = new longArray(num);
        for (int i = 0; i < num; i ++) {
            la.setitem(i, ids[i]);
        }
        return la;
    }
}
