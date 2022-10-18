#!/bin/bash

usage()
{
	echo "\
	create-aks.sh [OPTION...]
	--resource-group; Set a resource group name for AKS cluster
	--vnet-resource-group; vnet resource group to assign to AKS cluster
	--vnet-name; vnet name to assign to AKS cluster
	--subnet-name; subnet name to assign to AKS cluster
	--cluster-name; AKS cluster name
	--vm-size; AKS node vm size, should be DC-series, i.e. Standard_DC8ds_v3
	--node-count; AKS cluster initial node count
	--location; AKS's region
	--help; help
	" | column -t -s ";"
}

if [ "$#" -eq 0 ]; then
	usage
	exit 1
fi

while [ "$#" -gt 0 ]; do
        case $1 in
                --resource-group)
                        shift
                        if (( ! $# )); then
                            echo >&2 "$0: option $opt requires an argument."
                            exit 1
                        fi
                        BigDLResourceGroupName=$1
                        ;;
                --vnet-resource-group)
                        shift
                        if (( ! $# )); then
                            echo >&2 "$0: option $opt requires an argument."
                            exit 1
                        fi
                        VnetResourceGroupName=$1
                        ;;
                --vnet-name)
                        shift
                        if (( ! $# )); then
                            echo >&2 "$0: option $opt requires an argument."
                            exit 1
                        fi
                        VnetName=$1
                        ;;
                --subnet-name)
                        shift
                        if (( ! $# )); then
                            echo >&2 "$0: option $opt requires an argument."
                            exit 1
                        fi
                        SubnetName=$1
                        ;;
                --cluster-name)
                        shift
                        if (( ! $# )); then
                            echo >&2 "$0: option $opt requires an argument."
                            exit 1
                        fi
                        AKSClusterName=$1
                        ;;
                --vm-size)
                        shift
                        if (( ! $# )); then
                            echo >&2 "$0: option $opt requires an argument."
                            exit 1
                        fi
                        ConfVMSize=$1
                        ;;
                --node-count)
                        shift
                        if (( ! $# )); then
                            echo >&2 "$0: option $opt requires an argument."
                            exit 1
                        fi
                        NodeCount=$1
                        ;;
                --location)
                        shift
                        if (( ! $# )); then
                            echo >&2 "$0: option $opt requires an argument."
                            exit 1
                        fi
                        region=$1
                        ;;
                --help|-h)
                        usage
                        exit 0
                        ;;
                *)
                        echo >&2 "$0: unrecognized option $1."
                        usage
                        break
                        ;;
        esac

        shift
done
#region="eastus2"
echo "BigDLResourceGroupName: $BigDLResourceGroupName"
echo "AKSClusterName: $AKSClusterName"
echo "NodeCount: $NodeCount"
echo "ConfVMSize: $ConfVMSize"
echo "region: $region"

# Create Service Principle
appinfo=$(az ad sp create-for-rbac --query "[appId, password]" -o tsv)
readarray -d $'\n' -t app <<< $appinfo
APP_ID=${app[0]}
PASSWORD=${app[1]}
echo "APP_ID: $APP_ID"

# Assign your service princile to the VNet
VNET_ID=$(az network vnet show --resource-group $VnetResourceGroupName --name $VnetName --query id -o tsv)
echo "VNET_ID: $VNET_ID"
SUBNET_ID=$(az network vnet subnet show --resource-group $VnetResourceGroupName --vnet-name $VnetName --name $SubnetName --query id -o tsv)
echo "SUBNET_ID: $SUBNET_ID"

subscriptionId="$(az account list --query "[?isDefault].id" -o tsv)"
echo "subscriptionId: $subscriptionId"

az role assignment create --assignee $APP_ID --scope "/subscriptions/$subscriptionId/resourceGroups/$VnetResourceGroupName/providers/Microsoft.Network/virtualNetworks/$VnetName" --role "Network Contributor"

# Create aks cluster and enable confidential compute add-on
#ConfVMSize="Standard_DC8ds_v3"

az aks create \
    --resource-group $BigDLResourceGroupName \
    --name $AKSClusterName \
    --node-count $NodeCount \
    --node-vm-size $ConfVMSize \
    --network-plugin azure \
    --generate-ssh-keys \
    --vnet-subnet-id $SUBNET_ID \
    --enable-managed-identity \
    --enable-addons confcom \
    --location $region \
    --service-principal $APP_ID \
    --client-secret $PASSWORD

az aks addon update --addon confcom --name $AKSClusterName \
	--resource-group $BigDLResourceGroupName --enable-sgxquotehelper

