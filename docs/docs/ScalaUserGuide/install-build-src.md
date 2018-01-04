## **Download BigDL Source**

BigDL source code is available at [GitHub](https://github.com/intel-analytics/BigDL)

```bash
$ git clone https://github.com/intel-analytics/BigDL.git
```

By default, `git clone` will download the development version of BigDL, if you want a release version, you can use command `git checkout` to change the version. Available release versions is [BigDL releases](https://github.com/intel-analytics/BigDL/releases).


## **Setup Build Environment**

The following instructions are aligned with master code.

Maven 3 is needed to build BigDL, you can download it from the [maven website](https://maven.apache.org/download.cgi).

After installing Maven 3, please set the environment variable MAVEN_OPTS as follows:
```bash
$ export MAVEN_OPTS="-Xmx2g -XX:ReservedCodeCacheSize=512m"
```
When compiling with Java 7, you need to add the option “-XX:MaxPermSize=1G”. 


## **Build with script (Recommended)**

It is highly recommended that you build BigDL using the [make-dist.sh script](https://github.com/intel-analytics/BigDL/blob/master/make-dist.sh). And it will handle the MAVEN_OPTS variable.

Once downloaded, you can build BigDL with the following commands:
```bash
$ bash make-dist.sh
```
After that, you can find a `dist` folder, which contains all the needed files to run a BigDL program. The files in `dist` include:

* **dist/bin/bigdl.sh**: A script used to set up proper environment variables and launch the BigDL program.
* **dist/lib/bigdl-VERSION-jar-with-dependencies.jar**: This jar package contains all dependencies except Spark classes.
* **dist/lib/bigdl-VERSION-python-api.zip**: This zip package contains all Python files of BigDL.
* **dist/conf/spark-bigdl.conf**: This file contains necessary property configurations. ```Engine.createSparkConf``` will populate these properties, so try to use that method in your code. Or you need to pass the file to Spark with the "--properties-file" option. 

## **Build for Spark 2.0 and above**

The instructions above will build BigDL with Spark 1.5.x or 1.6.x (using Scala 2.10); to build for Spark 2.0 and above (which uses Scala 2.11 by default), pass `-P spark_2.x` to the `make-dist.sh` script:
```bash
$ bash make-dist.sh -P spark_2.x
```

It is highly recommended to use _**Java 8**_ when running with Spark 2.x; otherwise you may observe very poor performance.


## **Build for Scala 2.10 or 2.11**

By default, `make-dist.sh` uses Scala 2.10 for Spark 1.5.x or 1.6.x, and Scala 2.11 for Spark 2.0.x or 2.1.x. To override the default behaviors, you can pass `-P scala_2.10` or `-P scala_2.11` to `make-dist.sh` as appropriate.

---

## **Build native libs**

Note that the instructions above will skip the build of native library code, and pull the corresponding libraries from Maven Central. If you want to build the the native library code by yourself, follow the steps below:

 1.  Download and install [Intel Parallel Studio XE](https://software.intel.com//qualify-for-free-software/opensourcecontributor) in your Linux box.

 2.  Prepare build environment as follows:
 
```bash
    $ source <install-dir>/bin/compilervars.sh intel64
    $ source PATH_TO_MKL/bin/mklvars.sh intel64
```
    where the `PATH_TO_MKL` is the installation directory of the MKL.
    
 3. Full build
   
Clone BigDL as follows:
```bash
   git clone git@github.com:intel-analytics/BigDL.git --recursive 
```
For already cloned repos, just use:
```bash
   git submodule update --init --recursive 
```
If the Intel MKL is not installed to the default path `/opt/intel`, please pass your libiomp5.so's directory path to the `make-dist.sh` script:
```bash
   $ bash make-dist.sh -P full-build -DiompLibDir=<PATH_TO_LIBIOMP5_DIR> 
```
Otherwise, only pass `-P full-build` to the `make-dist.sh` script:
```bash
   $ bash make-dist.sh -P full-build
```

The defailts of building libraries on different platforms is at the end.
    
---
## **Build with Maven**

To build BigDL directly using Maven, run the command below:

```bash
$ mvn clean package -DskipTests
```
After that, you can find that the three jar packages in `PATH_To_BigDL`/target/, where `PATH_To_BigDL` is the path to the directory of the BigDL. 

Note that the instructions above will build BigDL with Spark 1.5.x or 1.6.x (using Scala 2.10) for Linux, and skip the build of native library code. Similarly, you may customize the default behaviors by passing the following parameters to maven:

 - `-P spark_2.x`: build for Spark 2.0 and above (using Scala 2.11). (Again, it is highly recommended to use _**Java 8**_ when running with Spark 2.0; otherwise you may observe very poor performance.)
 * `-P full-build`: full build
 * `-P scala_2.10` (or `-P scala_2.11`): build using Scala 2.10 (or Scala 2.11) 


---
## **Setup IDE**

We set the scope of spark related library to `provided` in pom.xml. The reason is that we don't want package spark related jars which will make bigdl a huge jar, and generally as bigdl is invoked by spark-submit, these dependencies will be provided by spark at run-time.

This will cause a problem in IDE. When you run applications, it will throw `NoClassDefFoundError` because the library scope is `provided`.

You can easily change the scopes by the `all-in-one` profile.

* In Intellij, go to View -> Tools Windows -> Maven Projects. Then in the Maven Projects panel, Profiles -> click "all-in-one". 


---
## **Build BigDL-core on different platforms**

### Environments Setup

For building BigDL-core, there should have

1. JDK 1.7+
2. maven
3. make
4. g++-7
5. Intel Parallel Studio
6. Git.

BigDL-core is a JNI project, `mkl2017-xeon-blas` needs MKL libraries with icc and `bigquant` needs g++-7. We use `maven` + `make` to control the build process where maven for java and make for c/c++ code.

#### CentOS

1. Build GCC-7.2
   1. Download GCC 7.2 source code
      1. `git clone https://github.com/gcc-mirror/gcc.git`
      2. `git checkout gcc-7_2_0-release`
   2. contrib/download_prerequisites
      1. base_url='http://gcc.gnu.org/pub/gcc/infrastructure/' # should change ftp to http because of proxy
   3. `./configure --prefix=/opt/gcc-7.2.0 --enable-languages=c,c++ --disable-multilib --disable-nls`
   4. `make -j4 && make install`
   5. `ln -s /opt/gcc-7.2.0 /opt/gcc`
2. binutils 2.29
   1. `wget https://ftp.gnu.org/gnu/binutils/binutils-2.29.tar.gz`
   2. `tar zxvf binutils-2.29.tar.gz -C /tmp/ && cd /tmp/binutils-2.29`
   3. `./configure --prefix=/opt/binutils-2.29`
   4. `make && make install`
   5. `ln -s /opt/binutils-2.29 /opt/binutils/`
3. Install Git
   1. `./configure --prefix=/opt/git-2.9.5`
   2. `make -j4 && make install`
   3. `ln -s /opt/git-2.9.5 /opt/git`
4. set environment variables
    ```
    export MAVEN_HOME=/opt/maven
    export PATH=$MAVEN_HOME/bin:$PATH
    export MAVEN_OPTS="-Xmx28g -Xss10M -XX:ReservedCodeCacheSize=512m -XX:MaxPermSize=128m"
    
    GCC_7_HOME=/opt/gcc
    LIBDIR=${GCC_7_HOME}/lib/../lib64
    export LD_LIBRARY_PATH=${LIBDIR}:${LD_LIBRARY_PATH}
    export LIBRARY_PATH=${LIBDIR}:${LIBRARY_PATH}
    export LD_RUN_PATH=${LIBDIR}:${LD_RUN_PATH}
    export PATH=${GCC_7_HOME}/bin/:${PATH}
    export C_INCLUDE_PATH=/opt/gcc/include/:${C_INCLUDE_PATH}
    export CPLUS_INCLUDE_PATH=/opt/gcc/include/:${CPLUS_INCLUDE_PATH}
    
    GIT_HOME=/opt/git
    export PATH=${GIT_HOME}/bin:${PATH}
    
    BINUTILS_HOME=/opt/binutils
    export LD_LIBRARY_PATH=${BINUTILS_HOME}/lib:${LD_LIBRARY_PATH}
    export PATH=${BINUTILS_HOME}/bin:${PATH}
    ```

#### Ubuntu/Debian

1. Install g++-7

   ```
   sudo add-apt-repository ppa:jonathonf/gcc-7.1
   sudo apt-get update
   sudo apt-get install gcc-7 g++-7
   sudo apt-get install build-essential
   ```

2. Install Parallel Studio XE

#### Windows

1. Install Visual Studio 2015
2. Install Intel Parallel Studio XE 2018
3. Install MinGW: https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/7.2.0/threads-win32/sjlj/x86_64-7.2.0-release-win32-sjlj-rt_v5-rev0.7z

   1. Unzip it to C:\MinGW.
   2. Set the environment
   3. Copy ming32-make.exe to make.exe
   4. Copy g++ to g++-7
   5. Open a cmd terminal and input `g++-7 -v` , should output like below,
      ```
      Using built-in specs.
      COLLECT_GCC=g++-7
      COLLECT_LTO_WRAPPER=C:/MinGW/bin/../libexec/gcc/x86_64-w64-mingw32/7.1.0/lto-wrapper.exe
      Target: x86_64-w64-mingw32
      Configured with: ../../../src/gcc-7.1.0/configure --host=x86_64-w64-mingw32 --build=x86_64-w64-mingw32 --target=x86_64-w64-ingw32 --prefix=/mingw64 --with-sysroot=/c/mingw710/x86_64-710-win32-sjlj-rt_v5-rev2/mingw64 --enable-shared --enable-static --enable-targets=all --enable-multilib --enable-languages=c,c++,fortran,lto --enable-libstdcxx-time=yes --enable-threads=win32 --enable-libgomp --enable-libatomic --enable-lto --enable-graphite --enable-checking=release --enable-fully-dynamic-string --enable-version-specific-runtime-libs --enable-libstdcxx-filesystem-ts=yes --enable-sjlj-exceptions --disable-libstdcxx-pch --disable-libstdcxx-debug --enable-bootstrap --disable-rpath --disable-win32-registry --disable-nls --disable-werror --disable-symvers --with-gnu-as --with-gnu-ld --with-arch-32=i686 --with-arch-64=nocona --with-tune-32=generic --with-tune-64=core2 --with-libiconv --with-system-zlib --with-gmp=/c/mingw710/prerequisites/x86_64-w64-mingw32-static --with-mpfr=/c/mingw710/prerequisites/x86_64-w64-mingw32-static --with-mpc=/c/mingw710/prerequisites/x86_64-w64-mingw32-static --with-isl=/c/mingw710/prerequisites/x86_64-w64-mingw32-static --with-pkgversion='x86_64-win32-sjlj-rev2, Built by MinGW-W64 project' --with-bugurl=https://sourceforge.net/projects/mingw-w64 CFLAGS='-O2 -pipe -fno-ident -I/c/mingw710/x86_64-710-win32-sjlj-rt_v5-rev2/mingw64/opt/include -I/c/mingw710/prerequisites/x86_64-zlib-static/include -I/c/mingw710/prerequisites/x86_64-w64-mingw32-static/include' CXXFLAGS='-O2 -pipe -fno-ident -I/c/mingw710/x86_64-710-win32-sjlj-rt_v5-rev2/mingw64/opt/include -I/c/mingw710/prerequisites/x86_64-zlib-static/include -I/c/mingw710/prerequisites/x86_64-w64-mingw32-static/include' CPPFLAGS=' -I/c/mingw710/x86_64-710-win32-sjlj-rt_v5-rev2/mingw64/opt/include -I/c/mingw710/prerequisites/x86_64-zlib-static/include -I/c/mingw710/prerequisites/x86_64-w64-mingw32-static/include' LDFLAGS='-pipe -fno-ident -L/c/mingw710/x86_64-710-win32-sjlj-rt_v5-rev2/mingw64/opt/lib -L/c/mingw710/prerequisites/x86_64-zlib-static/lib -L/c/mingw710/prerequisites/x86_64-w64-mingw32-static/lib '
      Thread model: win32
      gcc version 7.1.0 (x86_64-win32-sjlj-rev2, Built by MinGW-W64 project)
      ```

#### macOS

1. Install Parallel Studio XE.
2. Install g++-7: `brew install gcc@7`

### Build & Deploy

We use maven profile to control the build process. For different platforms has different profiles.

| Platform | Profile | Command |
|-----|:--:|:--:|
| Linux | linux | `mvn clean package -P linux` |
| RedHat5 | rh5 | `mvn clean package -P rh5` |
| macOS | mac   | `mvn clean package -P mac` |
| Windows | win64 | `mvn clean package -P win64` |

There two ways to deploy. We should use `mvn deploy -P deploy` at the end.
1. Build the jar on specific platform and deploy it. For example, we want to deploy bigquant of linux.
    ```
    mvn clean deploy -P 'linux' -pl 'bigquant/bigquant-java-x86_64-linux'
    ```
2. Copy the prebuilt libraries from every platform to a main machine, and deploy it.