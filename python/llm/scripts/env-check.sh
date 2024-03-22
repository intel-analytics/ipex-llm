#! /bin/bash  

check_python()
{
# Use this function to check whether Python exists
# and guarantee the Python version is higher than 3.9.x
# Otherwise, print message for users. 
  echo "-----------------------------------------------------------------"
  if python -V 2>&1 | awk '{print $2}' >/dev/null 2>&1
  then
    PY_VERSION=`python -V 2>&1 | awk '{print $2}'`
    echo -e "PYTHON_VERSION=$PY_VERSION"
    v1=`echo $PY_VERSION | awk -F '.' '{print $1}'`
    v2=`echo $PY_VERSION | awk -F '.' '{print $2}'`
    if [[ $(expr $v1) -ne 3 ]]  || [[ $(expr $v2) -lt 9 ]]
    then
      echo "Python Version must be higher than 3.9.x, please check python version. More details could be found in the README.md"
      retval="1"
    else
      retval="0"
    fi
  else
    echo "No Python found! Please use `conda create -n llm python=3.9` to create environment. More details could be found in the README.md"
    retval="1"
  fi
  return "$retval"
}


check_transformers()
{
  echo "-----------------------------------------------------------------"
  if python -c "import transformers; print(transformers.__version__)" >/dev/null 2>&1
  then
    VERSION=`python -c "import transformers; print(transformers.__version__)"`
    echo "transformers=$VERSION"
  else
    echo "Transformers is not installed. "
  fi
}

check_torch()
{
  echo "-----------------------------------------------------------------"
  if python -c "import torch; print(torch.__version__)" >/dev/null 2>&1
  then
    VERSION=`python -c "import torch; print(torch.__version__)"`
    echo "torch=$VERSION"
  else
    echo "PyTorch is not installed. "
  fi
}

check_ipex_llm()
{
  echo "-----------------------------------------------------------------"
  echo -n 'ipex-llm '
  pip show ipex-llm | grep Version:
}

check_cpu_info()
{
  echo "-----------------------------------------------------------------"
  echo "CPU Information: "
  lscpu | head -n 17
  
}

check_mem_info()
{
  echo "-----------------------------------------------------------------"
  cat /proc/meminfo | grep MemTotal
  
}

check_ulimit()
{
  echo "-----------------------------------------------------------------"
  echo "ulimit: "
  ulimit -a
}

check_os()
{
  echo "-----------------------------------------------------------------"
  echo "Operating System: "
  cat /etc/issue
}

check_env()
{
  echo "-----------------------------------------------------------------"
  echo "Environment Variable: "
  printenv
}

check_xpu_smi_install()
{
  echo "-----------------------------------------------------------------"
  if xpu-smi -h >/dev/null 2>&1
  then
    echo "xpu-smi is properly installed. " 
    return "0"
  else
    echo "xpu-smi is not installed. Please install xpu-smi according to README.md"
    return "1"
  fi
}

check_xpu_smi()
{
  echo "-----------------------------------------------------------------"
  xpu-smi discovery
}

check_ipex()
{
  echo "-----------------------------------------------------------------"
  if python -c "import intel_extension_for_pytorch as ipex; print(ipex.__version__)" >/dev/null 2>&1
  then
    VERSION=`python -c "import intel_extension_for_pytorch as ipex; print(ipex.__version__)"`
    echo "ipex=$VERSION"
  else
    echo "IPEX is not installed. "
  fi
}

main()
{
  # first guarantee correct python is installed. 
  check_python
  res=$?
  if [ $res != 0 ]
  then
    exit -1
  fi
  
  # check site packages version, such as transformers, pytorch, ipex_llm
  check_transformers
  check_torch
  check_ipex_llm
  check_ipex

  # verify hardware (how many gpu availables, gpu status, cpu info, memory info, etc.)
  check_cpu_info
  check_mem_info
  check_ulimit
  check_os
  check_env

  # check if xpu-smi and GPU is available. 
  check_xpu_smi_install
  res=$?
  if [ $res != 0 ]
  then
    exit -1
  else
    check_xpu_smi
  fi

  echo "-----------------------------------------------------------------"
}


main