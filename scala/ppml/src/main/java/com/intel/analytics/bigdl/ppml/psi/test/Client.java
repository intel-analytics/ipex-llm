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

import com.intel.analytics.bigdl.ppml.FLClient;
import com.intel.analytics.bigdl.ppml.psi.HashingUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.TimeUnit;


public class Client {
    private static final Logger logger = LoggerFactory.getLogger(Client.class);

    public static void main(String[] args) throws Exception {

        int max_wait = 20;
        // Example code for flClient
        int idSize = 11;
        // Quick lookup for the plaintext of hashed ids
        HashMap<String, String> data = HashingUtils.genRandomHashSet(idSize);
        HashMap<String, String> hashedIds = new HashMap<>();
        List<String> hashedIdArray;
        String salt;
        List<String> ids = new ArrayList<>(data.keySet());

        // Create a communication channel to the server,  known as a Channel. Channels are thread-safe
        // and reusable. It is common to create channels at the beginning of your application and reuse
        // them until the application shuts down.

        FLClient flClient = new FLClient();
        try {
            flClient.build();
            // Get salt from Server
            salt = flClient.psiStub().getSalt();
            logger.debug("Client get Slat=" + salt);
            // Hash(IDs, salt) into hashed IDs
            hashedIdArray = HashingUtils.parallelToSHAHexString(ids, salt);
            for (int i = 0; i < ids.size(); i++) {
                hashedIds.put(hashedIdArray.get(i), ids.get(i));
            }
            logger.debug("HashedIDs Size = " + hashedIds.size());
            flClient.psiStub().uploadSet(hashedIdArray);
            List<String> intersection;

            while (max_wait > 0) {
                intersection = flClient.psiStub().downloadIntersection();
                if (intersection == null) {
                    logger.info("Wait 1000ms");
                    Thread.sleep(1000);
                } else {
                    logger.info("Intersection successful. Intersection's size is " + intersection.size() + ".");
                    break;
                }
                max_wait--;
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally{
            // ManagedChannels use resources like threads and TCP connections. To prevent leaking these
            // resources the channel should be shut down when it will no longer be used. If it may be used
            // again leave it running.
            flClient.getChannel().shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }
}


