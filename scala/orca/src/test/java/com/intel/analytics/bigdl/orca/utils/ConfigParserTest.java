/*
 * Copyright 2021 The Analytic Zoo Authors
 *
 * Licensed under the Apache License,  Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,  software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,  either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.utils;

import com.fasterxml.jackson.core.JsonProcessingException;
import org.junit.Assert;
import org.junit.Test;

public class ConfigParserTest {
    @Test
    public void testConfigParserFromString() throws JsonProcessingException {
        String testString = String.join("\n",
                "stringProp: abc",
                "intProp: 123",
                "boolProp: true");
        TestHelper testHelper = ConfigParser.loadConfigFromString(testString, TestHelper.class);
        Assert.assertEquals(testHelper.intProp, 123);
        Assert.assertEquals(testHelper.boolProp, true);
        Assert.assertEquals(testHelper.stringProp, "abc");
    }
    @Test
    public void testConfigParserFromStringWithEmptyBool() throws JsonProcessingException {
        String testString = String.join("\n",
                "stringProp: abc",
                "intProp: 123");
        TestHelper testHelper = ConfigParser.loadConfigFromString(testString, TestHelper.class);
        Assert.assertEquals(testHelper.intProp, 123);
        Assert.assertEquals(testHelper.boolProp, false);
        Assert.assertEquals(testHelper.stringProp, "abc");
    }
    @Test
    public void testConfigParserFromStringWithEmptyString() throws JsonProcessingException {
        String testString = String.join("\n",
                "boolProp: true",
                "intProp: 123");
        TestHelper testHelper = ConfigParser.loadConfigFromString(testString, TestHelper.class);
        Assert.assertEquals(testHelper.intProp, 123);
        Assert.assertEquals(testHelper.boolProp, true);
        Assert.assertEquals(testHelper.stringProp, null);
    }
    @Test
    public void testConfigParserFromStringWithExtra() throws JsonProcessingException {
        String testString = String.join("\n",
                "stringProp: abc",
                "intProp: 123",
                "invalidProp: 123");
        TestHelper testHelper = ConfigParser.loadConfigFromString(testString, TestHelper.class);
        Assert.assertEquals(testHelper.intProp, 123);
        Assert.assertEquals(testHelper.boolProp, false);
        Assert.assertEquals(testHelper.stringProp, "abc");
    }

}

