set -x
export COUNTRY_NAME=cn
export CITY_NAME=shanghai
export ORGANIZATION_NAME=intel
export COMMON_NAME=heyang
export EMAIL_ADDRESS=heyang.sun@intel.com
export SERVER_PASSWORD=1234qwer

openssl req -new -key private.pem -out csr.pem \
        -subj "/C=$COUNTRY_NAME/ST=$CITY_NAME/L=$CITY_NAME/O=$ORGANIZATION_NAME/OU=$ORGANIZATION_NAME/CN=$COMMON_NAME"
