# Configure the variables to be passed into the templates.
# Set the versions according to your images
export pccsIP=your_pccs_ip_to_use_as # This is an IP to be used as k8s service IP, and it should be an IP address that unused in your subnetwork
export pccsImageName=intelanalytics/pccs:0.3.0-SNAPSHOT
export apiKey=your_intel_pcs_server_subscription_key_obtained_through_web_registeration # Refer to README for obtaining pcs api key
# Please do not use the same real IPs as nodes
export httpsProxyUrl=your_usable_https_proxy_url
export countryName=your_country_name
export cityName=your_city_name
export organizaitonName=your_organizaition_name
export commonName=server_fqdn_or_your_name
export emailAddress=your_email_address
export serverPassword=your_server_password_to_use 

## Create k8s namespace and apply BigDL-PCCS
kubectl create namespace bigdl-pccs
envsubst < bigdl-pccs.yaml | kubectl apply -f -
