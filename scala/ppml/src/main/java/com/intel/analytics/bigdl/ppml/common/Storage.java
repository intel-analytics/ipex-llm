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



import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Storage object could store Tensors at server including both,
 * server data for clients to download, and all client data uploaded at this version
 */
public class Storage<T> {
    private Logger logger = LogManager.getLogger(getClass());
    public String name;
    public int version;
    public T serverData = null;
    public Map<String, T> clientData;
    public Storage (String name) {
        version = 0;
        this.name = name;
        clientData = new ConcurrentHashMap<>();
    }
    public void updateStorage(T data) {
        clientData.clear();
        serverData = data;
        version += 1;
        logger.info("Storage " + name + " of version: " + version + " aggregated.");
    }
    /**
     *
     * @return The size of data collection of each local node
     */
    public int size() {
        return clientData.size();
    }

   
}
