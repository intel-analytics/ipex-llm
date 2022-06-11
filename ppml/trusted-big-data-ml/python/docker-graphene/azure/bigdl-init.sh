#!/bin/bash

while [ "$#" -gt 0 ]; do
        case $1 in
                -abc)
                        shift
                        { [ "$#" -ne 0 ] && is_option_arg "$1"; } || { usage >&2; exit 1; }
                        OPT_ABC=$1
                        ;;
                -def)
                        shift
                        { [ "$#" -ne 0 ] && is_option_arg "$1"; } || { usage >&2; exit 1; }
                        OPT_DEF=$1
                        ;;
                -verbose)
                        OPT_VERBOSE=true
                        ;;
                *)
                        break
                        ;;
        esac

        shift
done
#
# create resource group for bigdl
BigDLResourceGroupName=
region="eastus2"
#
# disable network policies first on the subnet as required by private endpoint and private link
VnetResourceGroupName=
VnetName=
SubnetName=

# Create Service Principle
APP_ID=$( az ad sp create-for-rbac --query id -o tsv)

# Assign your service princile to the VNet

VNET_ID=$(az network vnet show --resource-group $VnetResourceGroupName --name $VnetName --query id -o tsv)
SUBNET_ID=$(az network vnet subnet show --resource-group $VnetResourceGroupName --vnet-name $VnetName --name $SubnetName --query id -o tsv)
az role assignment create --assignee $APP_ID --scope "/subscriptions/xxx/resourceGroups/$VnetResourceGroupName/providers/Microsoft.Network/virtualNetworks/$SubnetName" --role "Network Contributor"

# create aks cluster and enable confidential compute add-on
AKSClusterName=
ConfVMSize="Standard_DC8ds_v3"
NodeCount=

az aks create \
    --resource-group $BigDLResourceGroupName \
    --name $AKSClusterName \
    --node-count $NodeCount \
    --node-vm-size $ConfVMSize \
    --vm-set-type AvailabilitySet \
    --network-plugin azure \
    --service-cidr 10.0.0.0/16 \
    --dns-service-ip 10.0.0.10 \
    --pod-cidr 10.244.0.0/16 \
    --docker-bridge-address 172.17.0.1/16 \
    --vnet-subnet-id $SUBNET_ID \
    --service-principal <appId> \
    --enable-managed-identity \
	--enable-addons confcom