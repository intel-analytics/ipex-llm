#!/bin/bash

# Check environment variables
if [ -z "${SKOPEO_HOME}" ]; then
    echo "Please set SKOPEO_HOME environment variable"
    exit 1
fi

# parameters
usage(){
echo "\
`cmd` [OPTION...]
--input-image; it should be oci image pulled by skopeo, e.g, oci:alpine
--output-image; destination image, e.g, oci:alpine-encrypted or docker://docker-registory/bigdl-k8:encrypted
--help; help
" | column -t -s ";"
}

while [ "$#" -gt 0 ]; do
        case $1 in
                --input-image)
                        shift
                        if (( ! $# )); then
                            echo >&2 "$0: option $opt requires an argument."
                            exit 1
                        fi
                        InputImage=$1
                        ;;
		--output-image)
                        shift
                        if (( ! $# )); then
                            echo >&2 "$0: option $opt requires an argument."
                            exit 1
                        fi
                        OutputImage=$1
                        ;;
		--encrypt-key)
                        shift
                        if (( ! $# )); then
                            echo >&2 "$0: option $opt requires an argument."
                            exit 1
                        fi
                        EncryptKey=$1
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

echo $InputImage
echo $OutputImage

${SKOPEO_HOME}/bin/skopeo copy --insecure-policy --encryption-key provider:attestation-agent:$EncryptKey $InputImage $OutputImage 
