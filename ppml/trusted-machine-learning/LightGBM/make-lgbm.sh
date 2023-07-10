#!/bin/bash
LGBM_NETWORK_MODE_RUNTIME=${LGBM_NETWORK_MODE_RUNTIME}
LGBM_NETWORK_MODE_BUILD=${LGBM_NETWORK_MODE_BUILD}
echo "LGBM_NETWORK_MODE_RUNTIME: ${LGBM_NETWORK_MODE_RUNTIME}"
echo "LGBM_NETWORK_MODE_BUILD: ${LGBM_NETWORK_MODE_BUILD}"

# For image build
if [ "${LGBM_NETWORK_MODE_BUILD}" == "" ] || [ "${LGBM_NETWORK_MODE_BUILD}" == "SSL_OR_PLAIN" ]
then
   echo "No mode assigned to LGBM network, use default SSL..."
   MAKE_FLAG=SSL
fi

if [ "${LGBM_NETWORK_MODE_RUNTIME}" == "" ] || [ "${LGBM_NETWORK_MODE_RUNTIME}" == "SSL_OR_PLAIN" ] && [ "${LGBM_NETWORK_MODE_BUILD}" != "" ]
then
   MAKE_FLAG=${LGBM_NETWORK_MODE_BUILD}
fi

# For user convenience to switch at runtime
if [ "${LGBM_NETWORK_MODE_RUNTIME}" != "" ]
then
   if [ "${LGBM_NETWORK_MODE_RUNTIME}" != "${LGBM_NETWORK_MODE_BUILD}" ]
   then
      echo "LGBM runtime network mode is different to the mode when building image, going to reset..."
      MAKE_FLAG=${LGBM_NETWORK_MODE_RUNTIME}
   else
      echo "LGBM has been build up, no need to make again."
      exit 0
   fi
fi

# Reset LGBM network mode by rebuilding the project
if [ -d "/ppml/LightGBM/build" ]
then
    rm -r /ppml/LightGBM/build
fi

cd /ppml/LightGBM
mkdir build
cd build
if [ "${MAKE_FLAG}" == "SSL" ]
then
   cmake -DUSE_SSL=1 .. # secure LGBM use ssl socket for communication
elif [ "${MAKE_FLAG}" == "PLAIN" ]
then
   cmake .. # use plain tcp socket
else
   echo "Wrong LGBM network mode is set: ${MAKE_FLAG}"
   exit -1
fi
make -j4
