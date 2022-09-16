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
import com.intel.test.TestValidationgObjectInputStream;
import org.junit.Test;
import java.io.InvalidClassException;

public class EncodeUtilsTest {

    @Test(expected = InvalidClassException.class)
    public void testBytesToObjNotAccept() throws InvalidClassException {
        // not accept class
        TestValidationgObjectInputStream test = new TestValidationgObjectInputStream();
        System.out.println(test.name);
        byte[] bytes = EncodeUtils.objToBytes(test);
        //should return null
        Object result = EncodeUtils.bytesToObj(bytes);
        if (result == null) {
            throw new InvalidClassException("objToBytes fail");
        }
    }

    @Test
    public void testBytesToObjAccept () throws InvalidClassException {
        // accept class
        String test = "test";
        System.out.println(test);
        byte[] bytes = EncodeUtils.objToBytes(test);
        //should not return null
        Object result = EncodeUtils.bytesToObj(bytes);
        if (result == null) {
            throw new InvalidClassException("accept class objToBytes fail");
        }
    }
}


