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

package com.intel.analytics.bigdl.friesian;

import com.intel.analytics.bigdl.friesian.serving.utils.EncodeUtils;
import com.intel.test.TestValidatingObjectInputStream;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;


public class EncodeUtilsTest {

    @Test
    public void testBytesToObjNotAccept() {
        // not accept class
        TestValidatingObjectInputStream test = new TestValidatingObjectInputStream();
        System.out.println(test.name);
        byte[] bytes = EncodeUtils.objToBytes(test);
        // should return null
        Object result = EncodeUtils.bytesToObj(bytes);
        assertNull(result, "objToBytes didn't fail as expected");
    }

    @Test
    public void testBytesToObjAccept() {
        // accept class
        String test = "test";
        System.out.println(test);
        byte[] bytes = EncodeUtils.objToBytes(test);
        // should not return null
        Object result = EncodeUtils.bytesToObj(bytes);
        assertNotNull(result, "objToBytes failed");
    }
}


