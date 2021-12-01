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

package com.intel.analytics.bigdl.ppml.psi.test;

import com.intel.analytics.bigdl.ppml.algorithms.PSI;
import com.intel.analytics.bigdl.ppml.vfl.VflContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.TimeUnit;


/**
 * For benchmark test only
 */
public class BenchmarkClient {
    private static final Logger logger = LoggerFactory.getLogger(BenchmarkClient.class);

    public static void main(String[] args) throws Exception {
        String taskID;
        int idSize;
        int startNum;
        // Number of arguments to be passed.
        if (args.length == 0) {
            logger.info("No argument passed, using default parameters.");
            taskID = "taskID";
            idSize = 10000;
            startNum = 0;
        } else {
            if (args.length != 3) {
                throw new Error("args length should be 3, taskID, idSize, startNum");
            }
            taskID = args[0];
            idSize = Integer.parseInt(args[1]);
            startNum = Integer.parseInt(args[2]);
        }
        logger.info("TaskID is: " + taskID);
        logger.info("id size: " + idSize + ", start num: " + startNum);

        // Example code for flClient
        // Quick lookup for the plaintext of hashed ids
        List<String> ids = new ArrayList<String>(idSize);
        long stproduce = System.currentTimeMillis();
        for (int i = startNum; i < idSize; i++) {
            ids.add(i-startNum, String.valueOf(i));
        }
        long etproduce = System.currentTimeMillis();
        logger.info("### Time of producing data: " + (etproduce - stproduce) + " ms ###");
        List<String> hashedIdArray;
        String salt;
        VflContext.initContext(null);
        PSI psi = new PSI();
        try {
            // Get salt from Server
            salt = psi.getSalt();
            logger.info("Client get Slat=" + salt);
            // Hash(IDs, salt) into hashed IDs

            long supload = System.currentTimeMillis();
            psi.uploadSet(ids, salt);

            long eupload = System.currentTimeMillis();
            logger.info("### Time of upload data: " + (eupload - supload) + " ms ###");
            logger.info("upload hashed id successfully");
            List<String> intersection = null;
            
            long sdownload = System.currentTimeMillis();
            psi.downloadIntersection(Integer.MAX_VALUE, 3000);

            long edownload = System.currentTimeMillis();
            logger.info("### Time of download data: " + (edownload - sdownload) + " ms ###");
            logger.info("Intersection successful. Total id(s) in intersection is " + intersection.size());

        } finally {
            // ManagedChannels use resources like threads and TCP connections. To prevent leaking these
            // resources the channel should be shut down when it will no longer be used. If it may be used
            // again leave it running.

            psi.close();
        }
    }
}


