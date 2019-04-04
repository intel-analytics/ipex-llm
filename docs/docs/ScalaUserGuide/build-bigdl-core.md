BigDL-core provides a lot of native methods including math processing routines,
quantized operation and so on. It will increase the performance of neural
networks. The math processing routines will call [Intel MKL](https://software.intel.com/en-us/mkl),
including gemm, add operations and so on.

BigDL-core is a JNI project, `mkl2017-xeon-blas` needs MKL libraries with icc and 
`bigquant` needs g++-7. We use `maven` + `make` to control the build process where
maven for java and make for c/c++ code. For building BigDL-core, there should have

+ JDK 1.7+
+ maven
+ make
+ g++-7
+ [Intel Parallel Studio](https://software.intel.com/en-us/parallel-studio-xe/choose-download)
+ Git.

## **Download BigDL-core Source**

BigDL-core source code is available at [GitHub](https://github.com/intel-analytics/BigDL)

```bash
$ git clone https://github.com/intel-analytics/BigDL-core.git
```

By default, `git clone` will download the development version of BigDL-core, if you want a release version, you can use command `git checkout` to change the version. 

## **Environment on CentOS** (after 6.8 final)

You should build the environment with CentOS 6.8 final. BigQuant depends the GCC 7.2,
and it's hard to find the a GCC 7.2 repo or rpm package that supports this version
OS. So you should build it from source. Another thing you should pay attention is,
you should build a new version binutils because the old version doesn't support to
assemble the code generated from GCC 7.2.

* **Build GCC-7.2**

```bash
git clone https://github.com/gcc-mirror/gcc.git
git checkout gcc-7_2_0-release
./configure --prefix=/opt/gcc-7.2.0 --enable-languages=c,c++ --disable-multilib --disable-nls
make -j4 && make install
ln -s /opt/gcc-7.2.0 /opt/gcc
```

* **Build binutils 2.29**

```bash
wget https://ftp.gnu.org/gnu/binutils/binutils-2.29.tar.gz
tar zxvf binutils-2.29.tar.gz -C /tmp/ && cd /tmp/binutils-2.29
configure --prefix=/opt/binutils-2.29
make && make install
ln -s /opt/binutils-2.29 /opt/binutils/
```

* **Install Git**

```bash
./configure --prefix=/opt/git-2.9.5
make -j4 && make install
ln -s /opt/git-2.9.5 /opt/git
```

* **Install Intel Parallel Studio XE**

* **Set environment variables**

```bash
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

## **Environment on Ubuntu/Debian**

* **Install g++-7**

```bash
sudo add-apt-repository ppa:jonathonf/gcc-7.1
sudo apt-get update
sudo apt-get install gcc-7 g++-7
sudo apt-get install build-essential
```

* **Install Intel Parallel Studio XE**

## **Environment on Windows 10**

For this platform, you should install Visual Studio 2015. We have not built
successfully with much newer VS. When you do the installation, you need only
forllow the default steps and click *next*.

Please pay attention, the MinGW must follows the link below. Otherwise it maybe
can't work. By default, the `make` names with `ming32-make.exe`, and `g++-7` names with `g++`.
You should change the name at last as follows.

1. **Install Visual Studio 2015**
2. **Install Intel Parallel Studio XE 2018**
3. **Install [MinGW](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/7.2.0/threads-win32/sjlj/x86_64-7.2.0-release-win32-sjlj-rt_v5-rev0.7z)**
4. **Copy `ming32-make.exe` to `make.exe`**
5. **Copy `g++` to `g++-7`**

At the end, open a cmd terminal and input `g++-7 -v` , should output like below,
      
```bash
Using built-in specs.
COLLECT_GCC=g++-7
COLLECT_LTO_WRAPPER=C:/MinGW/bin/../libexec/gcc/x86_64-w64-mingw32/7.1.0/lto-wrapper.exe
Target: x86_64-w64-mingw32
Configured with: ../../../src/gcc-7.1.0/configure --host=x86_64-w64-mingw32 --build=x86_64-w64-mingw32 --target=x86_64-w64-ingw32 --prefix=/mingw64 --with-sysroot=/c/mingw710/x86_64-710-win32-sjlj-rt_v5-rev2/mingw64 --enable-shared --enable-static --enable-targets=all --enable-multilib --enable-languages=c,c++,fortran,lto --enable-libstdcxx-time=yes --enable-threads=win32 --enable-libgomp --enable-libatomic --enable-lto --enable-graphite --enable-checking=release --enable-fully-dynamic-string --enable-version-specific-runtime-libs --enable-libstdcxx-filesystem-ts=yes --enable-sjlj-exceptions --disable-libstdcxx-pch --disable-libstdcxx-debug --enable-bootstrap --disable-rpath --disable-win32-registry --disable-nls --disable-werror --disable-symvers --with-gnu-as --with-gnu-ld --with-arch-32=i686 --with-arch-64=nocona --with-tune-32=generic --with-tune-64=core2 --with-libiconv --with-system-zlib --with-gmp=/c/mingw710/prerequisites/x86_64-w64-mingw32-static --with-mpfr=/c/mingw710/prerequisites/x86_64-w64-mingw32-static --with-mpc=/c/mingw710/prerequisites/x86_64-w64-mingw32-static --with-isl=/c/mingw710/prerequisites/x86_64-w64-mingw32-static --with-pkgversion='x86_64-win32-sjlj-rev2, Built by MinGW-W64 project' --with-bugurl=https://sourceforge.net/projects/mingw-w64 CFLAGS='-O2 -pipe -fno-ident -I/c/mingw710/x86_64-710-win32-sjlj-rt_v5-rev2/mingw64/opt/include -I/c/mingw710/prerequisites/x86_64-zlib-static/include -I/c/mingw710/prerequisites/x86_64-w64-mingw32-static/include' CXXFLAGS='-O2 -pipe -fno-ident -I/c/mingw710/x86_64-710-win32-sjlj-rt_v5-rev2/mingw64/opt/include -I/c/mingw710/prerequisites/x86_64-zlib-static/include -I/c/mingw710/prerequisites/x86_64-w64-mingw32-static/include' CPPFLAGS=' -I/c/mingw710/x86_64-710-win32-sjlj-rt_v5-rev2/mingw64/opt/include -I/c/mingw710/prerequisites/x86_64-zlib-static/include -I/c/mingw710/prerequisites/x86_64-w64-mingw32-static/include' LDFLAGS='-pipe -fno-ident -L/c/mingw710/x86_64-710-win32-sjlj-rt_v5-rev2/mingw64/opt/lib -L/c/mingw710/prerequisites/x86_64-zlib-static/lib -L/c/mingw710/prerequisites/x86_64-w64-mingw32-static/lib '
Thread model: win32
gcc version 7.1.0 (x86_64-win32-sjlj-rev2, Built by MinGW-W64 project)
```

## **Environment on macOS (after Sierra / 10.12.1)**

On macOS, you can install Intel Parallell Studio XE as normal. And for gcc, you
can use `brew`.

1. **Install Intel Parallel Studio XE.**
2. **Install g++-7:** `brew install gcc@7`.

## **Build and Deploy**

We use maven profile to control the build process. For different platforms has different profiles.

| Platform      | Profile | Command                      |
| -----         | :--:    | :--:                         |
| Linux         | linux   | `mvn clean package -P linux` |
| RedHat5       | rh5     | `mvn clean package -P rh5`   |
| macOS         | mac     | `mvn clean package -P mac`   |
| Windows       | win64   | `mvn clean package -P win64` |
| All Platforms | -       | `mvn clean package`          |

Note, the default command whithout any profile will generate a jar package which should contain
all libraries of different OS. But it requires you put the relevant libraries into the folder.
For example, for biquant, you should put the .so, .dylib, .dll to `bigquant/bigquant-java-<os>/target/clases`.

There two ways to deploy. You can use `mvn deploy -P deploy` at the end.

* Copy the prebuilt libraries from every platform to a main machine, and deploy it.
* Build the jar on specific platform and deploy it. For example, you want to deploy bigquant of linux.
  
```
mvn clean deploy -P 'linux' -pl 'bigquant/bigquant-java-x86_64-linux'
```
