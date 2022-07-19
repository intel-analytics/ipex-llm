package com.intel.analytics.bigdl.ppml;
import java.io.*;
import java.lang.*;
import java.util.*;

public class ProcessBuilderTest
{
    public static void main( String[] args ) throws IOException, InterruptedException
    {   System.out.println("Hello, World");
        ProcessBuilder pb = new ProcessBuilder(
            "bash", "-c", 
            "ls -al /;touch /test.txt;ls -al / | grep test.txt;chmod 777 /test.txt;ls -al / | grep test.txt;rm /test.txt;ls -al /");
        pb.inheritIO();
        Process process = pb.start();
        process.waitFor();
    }
}

