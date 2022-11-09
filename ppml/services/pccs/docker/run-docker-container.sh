export PCCS_IMAGE_NAME=your_pccs_image_name
export PCCS_IMAGE_VERSION=your_pccs_image_version
export PCCS_CONTAINER_NAME=your_pccs_container_name_to_run
export PCCS_PORT=pccs_port_to_use
export API_KEY=your_intel_pcs_server_subscription_key_obtained_through_web_registeration
export HTTPS_PROXY_URL=your_usable_https_proxy_url
# The above parameters are used to init pccs server
export COUNTRY_NAME=your_country_name
export CITY_NAME=your_city_name
export ORGANIZATION_NAME=your_organizaition_name
export COMMON_NAME=server_fqdn_or_your_name
export EMAIL_ADDRESS=your_email_address
export PASSWORD=your_server_password_to_use

docker run -itd \
--net=host \
--name $PCCS_CONTAINER_NAME \
-e PCCS_PORT=$PCCS_PORT \
-e API_KEY=$API_KEY \
-e HTTPS_PROXY_URL=$HTTPS_PROXY_URL \
-e COUNTRY_NAME=$COUNTRY_NAME \
-e CITY_NAME=$CITY_NAME \
-e ORGANIZATION_NAME==$ORGANIZATION_NAME \
-e COMMON_NAME=$COMMON_NAME \
-e EMAIL_ADDRESS=$EMAIL_ADDRESS \
-e PASSWORD=$PASSWORD \
-d $PCCS_IMAGE_NAME:$PCCS_IMAGE_VERSION
