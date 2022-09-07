/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.bigdl.ppml.fl.common;

import com.intel.analytics.bigdl.ppml.fl.base.DataHolder;
import com.intel.analytics.bigdl.ppml.fl.base.StorageHolder;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.HashMap;
import java.util.Map;


public abstract class Aggregator {
    /**
     * aggregateTypeMap is a map to map to simplify the operations of the storage
     * it maps the enum type: TRAIN, EVAL, PREDICT to corresponded storage
     */
    private Logger logger = LogManager.getLogger(getClass());
    public Map<FLPhase, StorageHolder> aggregateTypeMap;

    protected String returnMessage = "";

    public Aggregator() {
        aggregateTypeMap = new HashMap<>();
        initStorage();
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


    abstract public void initStorage();

    protected Integer clientNum;


    public abstract void aggregate(FLPhase flPhase);

    public void putClientData(FLPhase flPhase, int clientUUID, int version, DataHolder dataHolder)
            throws IllegalArgumentException, InterruptedException {
        StorageHolder storageHolder = aggregateTypeMap.get(flPhase);
        synchronized (this) {
            storageHolder.putClientData(clientUUID, dataHolder);
            logger.info(clientUUID + " uploaded. Server received: " + flPhase + " " +
                    storageHolder.getClientDataSize() + "/" + clientNum + ", version: " + version);
            if (storageHolder.getClientDataSize() >= clientNum) {
                logger.debug("Server start aggregate: " + flPhase);
                aggregate(flPhase);
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
