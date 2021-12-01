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

package com.intel.analytics.bigdl.ppml.common;

import com.intel.analytics.bigdl.ppml.generated.FLProto.Table;
import org.apache.log4j.Logger;

import java.util.HashMap;
import java.util.Map;

import static com.intel.analytics.bigdl.ppml.common.FLPhase.*;


public abstract class Aggregator<T> {
    /**
     * aggregateTypeMap is a map to map to simplify the operations of the storage
     * it maps the enum type: TRAIN, EVAL, PREDICT to corresponded storage
     */
    private Logger logger = Logger.getLogger(getClass());
    public Map<FLPhase, Storage<T>> aggregateTypeMap;

    public Aggregator() {
        aggregateTypeMap = new HashMap<>();
        aggregateTypeMap.put(TRAIN, trainStorage);
        aggregateTypeMap.put(EVAL, evalStorage);
        aggregateTypeMap.put(PREDICT, predictStorage);
        initStorage();
    }

    public void setClientNum(Integer clientNum) {
        this.clientNum = clientNum;
    }

    public Storage<T> trainStorage;
    public Storage<T> evalStorage;
    public Storage<T> predictStorage;

    public void initStorage() {
        trainStorage = new Storage<>("train");
        evalStorage = new Storage<>("eval");
        predictStorage = new Storage<>("predict");
    }

    protected Integer clientNum;
    public abstract void aggregate(FLPhase flPhase);

    public Storage<T> getServerData(FLPhase type) {
        Storage<T> storage = null;
        switch (type) {
            case TRAIN: storage = trainStorage; break;
            case EVAL: storage = evalStorage; break;
            case PREDICT: storage = predictStorage; break;
            default: break;
        }
        return storage;
    }
    public <T> void putClientData(FLPhase type, String clientUUID, int version, T data)
            throws IllegalArgumentException {
        Storage storage = getServerData(type);
        storage.put(clientUUID, data);

        // Aggregate when buffer is full
        if (storage.size() >= clientNum) {
            aggregate(type);
        }
    }



}
