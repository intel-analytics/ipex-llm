#!/bin/bash
set -e
set -x

usage(){
echo "\
`cmd` [OPTION...]
--registry-name; it should be host domain name
--help; help
" | column -t -s ";"
}

while [ "$#" -gt 0 ]; do
        case $1 in
                --registry-name)
                        shift
                        if (( ! $# )); then
                            echo >&2 "$0: option $opt requires an argument."
                            exit 1
                        fi
                        RegistryName=$1
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

echo $RegistryName

# Part1: generate certificate for docker registry
echo "generate certificate for docker registry"
openssl req \
-newkey rsa:4096 -nodes -sha256 -keyout certs/domain.key \
-addext "subjectAltName = DNS:$RegistryName" \
-x509 -days 365 -out certs/domain.crt

# Part2: create docker registry container
echo "create docker registry"
docker run -d \
--restart=always \
--name registry \
-v "$(pwd)"/certs:/certs \
-e REGISTRY_HTTP_ADDR=0.0.0.0:443 \
-e REGISTRY_HTTP_TLS_CERTIFICATE=/certs/domain.crt \
-e REGISTRY_HTTP_TLS_KEY=/certs/domain.key \
-p 443:443 \
registry:2

echo "registry is created successfully"
