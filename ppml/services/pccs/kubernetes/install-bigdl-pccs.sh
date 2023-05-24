# Configure the variables to be passed into the templates.
# Set the versions according to your images
export pccsIP=your_pccs_ip_to_use_as # This is an IP to be used as k8s service IP, and it should be an IP address that unused in your subnetwork
export pccsImageName=intelanalytics/pccs:0.3.0-SNAPSHOT
export apiKey=your_intel_pcs_server_subscription_key_obtained_through_web_registeration # Refer to README for obtaining pcs api key
export userPassword=a_password_for_pccs_user
export adminPassword=a_password_for_pccs_admin
# Please do not use the same real IPs as nodes
export httpsProxyUrl=your_usable_https_proxy_url
export countryName=your_country_name
export cityName=your_city_name
export organizaitonName=your_organizaition_name
export commonName=server_fqdn_or_your_name
export emailAddress=your_email_address
export httpsCertPassword=your_https_cert_password_to_use 

if [[ "$pccsImageName" == "your_pccs_image_name" ]] || \
   [[ "$pccsIP" == "your_pccs_ip_to_use_as" ]]
then
    echo "modify pccs container configurations in the script please!"
    exit -1
fi

if [[ "$apiKey" == "your_intel_pcs_server_subscription_key_obtained_through_web_registeration" ]]
then
    echo "register PCS and modify API_KEY in the script please!"
    exit -1
fi

if [[ "$httpsProxyUrl" == "your_usable_https_proxy_url" ]]
then
    echo "set proxy for pccs container please!"
    echo "if no proxy is needed in your container environment, set HTTPS_PROXY_URL= "" "
    exit -1
fi

if [[ "$userPassword" == "a_password_for_pccs_user" ]] || \
   [[ "$adminPassword" == "a_password_for_pccs_admin" ]] || \
   [[ "$httpsCertPassword" == "your_https_cert_password_to_use" ]]
then
    echo "modify password configurations in the script please!"
    echo "USER_PASSWORD for pccs user token, ADMIN_PASSWORD for pccs admin token and HTTPS_CERT_PASSWORD for https cert signing"
    exit -1
fi

if [[ "$countryName" == "your_country_name" ]] || \
   [[ "$cityName" == "your_city_name" ]] || \
   [[ "$organizaitonName" == "your_organizaition_name" ]] || \
   [[ "$emailAddress" == "your_email_address" ]] || \
   [[ "$commonName" == "server_fqdn_or_your_name" ]]
then
    echo "modify https cert configurations in the script please!"
    exit -1
fi

## Create k8s namespace and apply BigDL-PCCS
kubectl create namespace bigdl-pccs
envsubst < bigdl-pccs.yaml | kubectl apply -f -
