package com.intel.analytics.bigdl.ppml.attestation;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import sun.misc.BASE64Encoder;

import javax.crypto.Mac;
import javax.crypto.SecretKey;
import javax.crypto.spec.SecretKeySpec;
import javax.net.ssl.*;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.security.cert.X509Certificate;
import java.util.HashMap;

public class RegisterMrenclave {
    public  String appId;
    public String apiKey;
    public String baseUrl;
    public String mrEnclave;
    public String mrSigner;

    final static HostnameVerifier DO_NOT_VERIFY = (hostname, session) -> true;

    public static void main(String[] args) throws IOException, NoSuchAlgorithmException, InvalidKeyException {
        // read parameter
        RegisterMrenclave rs = new RegisterMrenclave();
        rs.appId = args[0];
        rs.apiKey = args[1];
        rs.baseUrl = args[2];
        rs.mrEnclave = args[3];
        rs.mrSigner = args[4];
        rs.registerMrenclave();
    }

    private void registerMrenclave() throws IOException, NoSuchAlgorithmException, InvalidKeyException {

        // Create a trust manager that does not validate certificate chains
        trustAllHosts();
        URL url = new URL(baseUrl + "/ehsm?Action=" + "UploadQuotePolicy");
        HttpsURLConnection https = (HttpsURLConnection) url.openConnection();
        https.setRequestMethod("POST");
        https.setConnectTimeout(6000);
        https.setDoOutput(true);
        https.setDoInput(true);
        https.setUseCaches(false);
        https.setInstanceFollowRedirects(true);
        https.setRequestProperty("Content-Type", "application/json");
        https.setHostnameVerifier(DO_NOT_VERIFY);
        https.connect();

        HashMap<String, Object> map = new HashMap();
        map.put("appid", appId);
        map.put("timestamp", String.valueOf(System.currentTimeMillis()));
        HashMap<String, String> map1 = new HashMap();
        map1.put("mr_enclave", mrEnclave);
        map1.put("mr_signer", mrSigner);
        map.put("payload", map1);

        //creat sign
        byte[] data = apiKey.getBytes("UTF-8");
        System.out.println(new String(data, "UTF-8"));
        SecretKey secretKey = new SecretKeySpec(data, "HmacSHA256");
        Mac mac = Mac.getInstance("HmacSHA256");
        mac.init(secretKey);

        String merge = "appid=" + appId + "&payload=mr_enclave=" + mrEnclave + "&mr_signer=" + mrSigner + "&timestamp=" + map.get("timestamp");
        byte[] sign = merge.getBytes("UTF-8");
        //sign to byte
        byte[] bytes = mac.doFinal(sign);
        final BASE64Encoder encoder = new BASE64Encoder();
        map.put("sign", encoder.encode(bytes));
        String mapJson = new Gson().toJson(map);


        DataOutputStream dos = new DataOutputStream(https.getOutputStream());
        dos.write(mapJson.getBytes("UTF-8"));
        dos.flush();
        dos.close();


        if (https.getResponseCode() == 200) {
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(https.getInputStream()));
            String rep = reader.readLine();
            //System.out.println(rep);
            JsonObject resJson = new JsonParser().parse(rep).getAsJsonObject();
            JsonObject result = resJson.getAsJsonObject("result");
            String policyId = result.get("policyId").toString().replace("\"", "");
            System.out.println("policy_Id " + policyId); // user for grep
        } else {
            System.out.println("register fail");
            return;
        }

    }

    private static void trustAllHosts() {

        // Create a trust manager that does not validate certificate chains
        TrustManager[] trustAllCerts = new TrustManager[] { new X509TrustManager() {

            public java.security.cert.X509Certificate[] getAcceptedIssuers() {
                return new java.security.cert.X509Certificate[] {};
            }

            public void checkClientTrusted(X509Certificate[] chain, String authType) {

            }

            public void checkServerTrusted(X509Certificate[] chain, String authType) {

            }
        } };

        // Install the all-trusting trust manager
        try {
            SSLContext sc = SSLContext.getInstance("TLS");
            sc.init(null, trustAllCerts, new java.security.SecureRandom());
            HttpsURLConnection.setDefaultSSLSocketFactory(sc.getSocketFactory());
        } catch (Exception e) {
            System.out.println("trustAllHosts is error");
            e.printStackTrace();
        }
    }
}
