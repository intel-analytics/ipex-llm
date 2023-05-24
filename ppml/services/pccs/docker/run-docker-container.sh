export PCCS_IMAGE_NAME=your_pccs_image_name
export PCCS_IMAGE_VERSION=your_pccs_image_version
export PCCS_CONTAINER_NAME=your_pccs_container_name_to_run
export PCCS_PORT=pccs_port_to_use
export API_KEY=your_intel_pcs_server_subscription_key_obtained_through_web_registeration
export HTTPS_PROXY_URL=your_usable_https_proxy_url
export USER_PASSWORD=a_password_for_pccs_user
export ADMIN_PASSWORD=a_password_for_pccs_admin
# The above parameters are used to init pccs server
export COUNTRY_NAME=your_country_name
export CITY_NAME=your_city_name
export ORGANIZATION_NAME=your_organizaition_name
export COMMON_NAME=server_fqdn_or_your_name
export EMAIL_ADDRESS=your_email_address
export HTTPS_CERT_PASSWORD=your_https_cert_password_to_use

if [[ "$PCCS_IMAGE_NAME" == "your_pccs_image_name" ]] || \
   [[ "$PCCS_IMAGE_VERSION" == "your_pccs_image_version" ]] || \
   [[ "$PCCS_CONTAINER_NAME" == "your_pccs_container_name_to_run" ]] || \
   [[ "$PCCS_PORT" == "pccs_port_to_use" ]]
then
    echo "modify pccs container configurations in the script please!"
    exit -1
fi

if [[ "$API_KEY" == "your_intel_pcs_server_subscription_key_obtained_through_web_registeration" ]]
then
    echo "register PCS and modify API_KEY in the script please!"
    exit -1
fi

if [[ "$HTTPS_PROXY_URL" == "your_usable_https_proxy_url" ]]
then
    echo "set proxy for pccs container please!"
    echo "if no proxy is needed in your container environment, set HTTPS_PROXY_URL= "" "
    exit -1
fi

if [[ "$USER_PASSWORD" == "a_password_for_pccs_user" ]] || \
   [[ "$ADMIN_PASSWORD" == "a_password_for_pccs_admin" ]] || \
   [[ "$HTTPS_CERT_PASSWORD" == "your_https_cert_password_to_use " ]]
then
    echo "modify password configurations in the script please!"
    echo "USER_PASSWORD for pccs user token, ADMIN_PASSWORD for pccs admin token and HTTPS_CERT_PASSWORD for https cert signing"
    exit -1
fi

if [[ "$COUNTRY_NAME" == "your_country_name" ]] || \
   [[ "$CITY_NAME" == "your_city_name" ]] || \
   [[ "$ORGANIZATION_NAME" == "your_organizaition_name" ]] || \
   [[ "$EMAIL_ADDRESS" == "your_email_address" ]] || \
   [[ "$COMMON_NAME" == "server_fqdn_or_your_name" ]]
then
    echo "modify https cert configurations in the script please!"
    exit -1
fi

docker run -itd \
--net=host \
--name $PCCS_CONTAINER_NAME \
-e PCCS_PORT=$PCCS_PORT \
-e API_KEY=$API_KEY \
-e HTTPS_PROXY_URL=$HTTPS_PROXY_URL \
-e USER_PASSWORD=$USER_PASSWORD \
-e ADMIN_PASSWORD=$ADMIN_PASSWORD \
-e COUNTRY_NAME=$COUNTRY_NAME \
-e CITY_NAME=$CITY_NAME \
-e ORGANIZATION_NAME==$ORGANIZATION_NAME \
-e COMMON_NAME=$COMMON_NAME \
-e EMAIL_ADDRESS=$EMAIL_ADDRESS \
-e HTTPS_CERT_PASSWORD=$HTTPS_CERT_PASSWORD \
-d $PCCS_IMAGE_NAME:$PCCS_IMAGE_VERSION
