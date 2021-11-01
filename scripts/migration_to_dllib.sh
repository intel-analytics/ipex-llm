# This script is used to automatically migrate existing projects which usd BigDL0.x or Analytics Zoo to use Bigdl-dllib. Please run this script if you want to migrate your existing BigdL0.x or zoo projects to BigDL2.x
# It will do inplace update. Please backup your folder before the migration. It may output "sed: no input files", that's normal and can be ingored.

if (( $# < 1)); then
  echo "Usage: migration_to_dllib.sh folder_need_to_be_migrated"
  exit -1
fi

dir=$1
grep -rl com.intel.analytics.bigdl.utils --include \*.scala ${dir} | xargs sed -i 's/com.intel.analytics.bigdl.utils/com.intel.analytics.bigdl.dllib.utils/g'
grep -rl com.intel.analytics.bigdl.optim --include \*.scala ${dir} | xargs sed -i 's/com.intel.analytics.bigdl.optim/com.intel.analytics.bigdl.dllib.optim/g'
grep -rl com.intel.analytics.bigdl.tensor --include \*.scala ${dir} | xargs sed -i 's/com.intel.analytics.bigdl.tensor/com.intel.analytics.bigdl.dllib.tensor/g'
grep -rl com.intel.analytics.bigdl.nn --include \*.scala ${dir} | xargs sed -i 's/com.intel.analytics.bigdl.nn/com.intel.analytics.bigdl.dllib.nn/g'
grep -rl com.intel.analytics.bigdl.models --include \*.scala ${dir} | xargs sed -i 's/com.intel.analytics.bigdl.models/com.intel.analytics.bigdl.dllib.models/g'
grep -rl com.intel.analytics.bigdl.dataset --include \*.scala ${dir} | xargs sed -i 's/com.intel.analytics.bigdl.dataset/com.intel.analytics.bigdl.dllib.feature.dataset/g'
grep -rl com.intel.analytics.bigdl.transform --include \*.scala ${dir} | xargs sed -i 's/com.intel.analytics.bigdl.transform/com.intel.analytics.bigdl.dllib.feature.transform/g'
grep -rl com.intel.analytics.zoo.feature --include \*.scala ${dir} | xargs sed -i 's/com.intel.analytics.zoo.feature/com.intel.analytics.bigdl.dllib.feature/g'
grep -rl com.intel.analytics.zoo.pipeline.api.keras --include \*.scala ${dir} | xargs sed -i 's/com.intel.analytics.zoo.pipeline.api.keras/com.intel.analytics.bigdl.dllib.keras/g'
grep -rl com.intel.analytics.zoo.common.NNContext --include \*.scala ${dir} | xargs sed -i 's/com.intel.analytics.zoo.common.NNContext/com.intel.analytics.bigdl.dllib.NNContext/g'
grep -rl com.intel.analytics.zoo.common --include \*.scala ${dir} | xargs sed -i 's/com.intel.analytics.zoo.common/com.intel.analytics.bigdl.dllib.common/g'
grep -rl com.intel.analytics.zoo.pipeline.nnframes --include \*.scala ${dir} | xargs sed -i 's/com.intel.analytics.zoo.pipeline.nnframes/com.intel.analytics.bigdl.dllib.nnframes/g'
                                               
grep -rl bigdl.dataset --include \*.py ${dir} | xargs sed -i 's/bigdl.dataset/bigdl.dllib.feature.dataset/g'
grep -rl bigdl.transform --include \*.py ${dir} | xargs sed -i 's/bigdl.transform/bigdl.dllib.feature.transform/g'
grep -rl bigdl.nn --include \*.py ${dir} | xargs sed -i 's/bigdl.nn/bigdl.dllib.nn/g'
grep -rl bigdl.optim --include \*.py ${dir} | xargs sed -i 's/bigdl.optim/bigdl.dllib.optim/g'
grep -rl bigdl.util --include \*.py ${dir} | xargs sed -i 's/bigdl.util/bigdl.dllib.utils/g'
grep -rl zoo.common --include \*.py ${dir} | xargs sed -i 's/zoo.common/bigdl.dllib.utils/g'
grep -rl bigdl.dllib.utils.nncontext --include \*.py ${dir} | xargs sed -i 's/bigdl.dllib.utils.nncontext/bigdl.dllib.nncontext/g'
grep -rl zoo.feature --include \*.py ${dir} | xargs sed -i 's/zoo.feature/bigdl.dllib.feature/g'
grep -rl zoo.pipeline.nnframes --include \*.py ${dir} | xargs sed -i 's/zoo.pipeline.nnframes/bigdl.dllib.nnframes/g'
grep -rl zoo.pipeline.api.keras --include \*.py ${dir} | xargs sed -i 's/zoo.pipeline.api.keras/bigdl.dllib.keras/g'
grep -rl zoo.pipeline.api.autograd --include \*.py ${dir} | xargs sed -i 's/zoo.pipeline.api.autograd/bigdl.dllib.keras.autograd/g'
