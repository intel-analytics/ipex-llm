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

import com.intel.analytics.bigdl.ppml.generated.FlBaseProto.Table;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.HashMap;
import java.util.Map;

import static com.intel.analytics.bigdl.ppml.common.FLPhase.*;


public abstract class Aggregator<T> {
    /**
     * aggregateTypeMap is a map to map to simplify the operations of the storage
     * it maps the enum type: TRAIN, EVAL, PREDICT to corresponded storage
     */
    private Logger logger = LogManager.getLogger(getClass());
    public Map<FLPhase, Storage<T>> aggregateTypeMap;
    protected Boolean hasReturn = false;
    protected String returnMessage = "";

    public Aggregator() {
        initStorage();
        aggregateTypeMap = new HashMap<>();
        aggregateTypeMap.put(TRAIN, trainStorage);
        aggregateTypeMap.put(EVAL, evalStorage);
        aggregateTypeMap.put(PREDICT, predictStorage);

    }

    public void setReturnMessage(String returnMessage) {
        this.returnMessage = returnMessage;
    }

    public String getReturnMessage() {
        return returnMessage;
    }

    public void setClientNum(Integer clientNum) {
        this.clientNum = clientNum;
    }

    public void setHasReturn(Boolean hasReturn) {
        this.hasReturn = hasReturn;
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
            throws IllegalArgumentException, InterruptedException {
        logger.debug(clientUUID + " getting data to update from server: " + type.toString());
        Storage storage = getServerData(type);
        checkVersion(storage.version, version);
        logger.debug(clientUUID + " version check pass, version: " + version);


        synchronized (this) {
            storage.clientData.put(clientUUID, data);
            logger.debug(clientUUID + " client data uploaded to server: " + type.toString());
            logger.debug("Server received data " + storage.size() + "/" + clientNum);
            if (storage.size() >= clientNum) {
                logger.debug("Server received all client data, start aggregate.");
                aggregate(type);
                notifyAll();
            } else {
                wait();
            }
        }

    }
    protected void checkVersion(int serverVersion, int clientVersion)
            throws IllegalArgumentException {
        if (serverVersion != clientVersion) {
            throw new IllegalArgumentException("Version miss match, got server version: " +
                    serverVersion + ", client version: " + clientVersion);
        }
    }

}
