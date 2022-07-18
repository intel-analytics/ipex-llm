package com.intel.analytics.bigdl.ppml;
import java.io.File;  
import java.io.IOException; 
import java.io.FileWriter;
import java.util.*;
import java.nio.file.*;
import java.nio.file.attribute.*;
public class FileSystemIO {
    public static void main(String[] args)  throws IOException, InterruptedException{
        try {
            // Create
            File file = new File("test.txt");
            if (file.createNewFile()) {
                System.out.println("File created: " + file.getName());
            } else {
                System.out.println("File already exists.");
            }
            //check
            ProcessBuilder pb = new ProcessBuilder("bash", "-c","ls -al");
            pb.inheritIO();
            Process process = pb.start();
            process.waitFor();
            //Write
            FileWriter myWriter = new FileWriter("test.txt");
            myWriter.write("Bigdl ppml");
            myWriter.close();
            //check
            pb = new ProcessBuilder("bash", "-c","cat test.txt | grep ppml");
            pb.inheritIO();
            process = pb.start();
            process.waitFor();
            //chmod
            Set<PosixFilePermission> perms = new HashSet<>();
            perms.add(PosixFilePermission.OWNER_READ);
            perms.add(PosixFilePermission.GROUP_WRITE);
            perms.add(PosixFilePermission.OTHERS_EXECUTE); 

            Files.setPosixFilePermissions(file.toPath(), perms);
            //check
            pb = new ProcessBuilder("bash", "-c","ls -al| grep test.txt");
            pb.inheritIO();
            process = pb.start();
            process.waitFor();
            //Delete
            if (file.delete()) { 
                System.out.println("Deleted: " + file.getName());
            } else {
                System.out.println("Failed to delete.");
            } 
            //check
            pb = new ProcessBuilder("bash", "-c","ls -al");
            pb.inheritIO();
            process = pb.start();
            process.waitFor();
        } catch (IOException e) {
            System.out.println("Error occurred in java execution.");
            e.printStackTrace();
        }
    }
}
