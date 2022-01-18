# Run PPML Demo in Graphene

## Before running code

### Build ppml-jar
run:
```bash
cd .. && mvn clean package -DskipTests -Pspark_3.x
mv target/bigdl-ppml-spark_3.1.2-0.14.0-SNAPSHOT-jar-with-dependencies.jar demo
cd demo
```

### Build Image
Modify your http_proxy in `build-image.sh` then run:
```bash
./build-image.sh
```

### Prepare key and password
Following these steps to prepare [key](http://10.112.231.51:18888/view/ZOO-PR/job/BigDL-PR-PPML-UTs-Validation/8/console) and [password](http://10.112.231.51:18888/view/ZOO-PR/job/BigDL-PR-PPML-UTs-Validation/8/console).Then modify these path and your local ip in `deploy-local-spark-sgx.sh`.

## Start container
run:
```bash
./deploy-local-spark-sgx.sh
sudo exec -it flDemo bash
./init.sh
```

## Start FLServer
In container run:
```bash
./runFlServer.sh
```

## Run HFL Demo
Open two new windows, run:
```bash
sudo exec -it flDemo bash
```
to enter the container, then in a window run:
```bash
./runHflClient1.sh
```
in another window run:
```bash
./runHflClient2.sh
```

## Run VFL Demo
Open two new windows, run:
```bash
sudo exec -it flDemo bash
```
to enter the container, then in a window run:
```bash
./runVflClient1.sh
```
in another window run:
```bash
./runVflClient2.sh
```

