package com.intel.analytics.bigdl.friesian;

import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity;
import com.intel.test.TestValidationgObjectInputStream;
import com.intel.analytics.bigdl.friesian.serving.utils.EncodeUtils;
import org.junit.Test;

import java.io.InvalidClassException;

import static org.junit.jupiter.api.Assertions.*;

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


