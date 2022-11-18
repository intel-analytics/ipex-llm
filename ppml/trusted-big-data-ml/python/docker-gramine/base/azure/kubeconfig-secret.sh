#!/bin/bash
kubectl config view --flatten --minify > kubeconfig.txt
kubectl create secret generic kubeconf --from-file=kubeconfig=./kubeconfig.txt