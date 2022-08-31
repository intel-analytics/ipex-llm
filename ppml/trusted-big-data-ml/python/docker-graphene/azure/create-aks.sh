#!/bin/bash

usage(){
echo "\
`cmd` [OPTION...]
--resource-group; Set a resource group name for AKS cluster
--vnet-resource-group; vnet resource group to assign to AKS cluster
--vnet-name; vnet name to assign to AKS cluster
--subnet-name; subnet name to assign to AKS cluster
--cluster-name; AKS cluster name
--vm-size; AKS node vm size, should be DC-series, i.e. Standard_DC8ds_v3
--node-count; AKS cluster initial node count
--help; help
" | column -t -s ";"
}

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
region="eastus2"

# Create Service Principle
APP_ID=$( az ad sp create-for-rbac --query id -o tsv)

# Assign your service princile to the VNet
VNET_ID=$(az network vnet show --resource-group $VnetResourceGroupName --name $VnetName --query id -o tsv)
SUBNET_ID=$(az network vnet subnet show --resource-group $VnetResourceGroupName --vnet-name $VnetName --name $SubnetName --query id -o tsv)
az role assignment create --assignee $APP_ID --scope "/subscriptions/xxx/resourceGroups/$VnetResourceGroupName/providers/Microsoft.Network/virtualNetworks/$SubnetName" --role "Network Contributor"

# Create aks cluster and enable confidential compute add-on
ConfVMSize="Standard_DC8ds_v3"

az aks create \
    --resource-group $BigDLResourceGroupName \
    --name $AKSClusterName \
    --node-count $NodeCount \
    --node-vm-size $ConfVMSize \
    --vm-set-type AvailabilitySet \
    --network-plugin azure \
    --generate-ssh-keys \
    --vnet-subnet-id $SUBNET_ID \
    --service-principal $APP_ID \
    --enable-managed-identity \
	--enable-addons confcom