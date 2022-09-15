package com.intel.analytics.bigdl.friesian;

import com.google.gson.JsonObject;
import com.intel.Test.TestValidationgObjectInputStream;
import com.intel.analytics.bigdl.friesian.serving.utils.EncodeUtils;
import org.junit.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.io.InvalidClassException;
public class EncodeUtilsTest {

    @Test
    public void testBytesToObj(){
        TestValidationgObjectInputStream test = new TestValidationgObjectInputStream();
        System.out.println(test.name);
        byte[] bytes = EncodeUtils.objToBytes(test);
        //should return null
        Object result = EncodeUtils.bytesToObj(bytes);
        assertNull(result);
    }
}


