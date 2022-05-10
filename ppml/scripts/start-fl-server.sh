BIGDL_PPML_JAR_PATH=$(find . -maxdepth 1 -name 'bigdl-ppml-*-jar-with-dependencies.jar')

while [ "$1" != "" ]; do
    case $1 in
        -c | --config_path )    shift
                                config_path=$1
                                ;;        
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

if [ -z "${config_path// }" ]; then
  config_path=ppml-conf.yaml
fi
java -cp $BIGDL_PPML_JAR_PATH com.intel.analytics.bigdl.ppml.fl.FLServer -c $config_path > fl-server.log &
