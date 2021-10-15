cd ${BASE}/python/serving
if [ -d "${BASE}/python/serving/dist" ]; then
   rm -r ${BASE}/python/serving/dist
fi
python setup.py bdist_wheel --universal

if [ -d "${BASE}/zoo/serving/build" ]; then
   echo "Removing serving/build"
   rm -r ${BASE}/zoo/serving/build
fi
if [ -d "${BASE}/zoo/serving/bigdl_serving.egg-info" ]; then
   echo "Removing serving/bigdl_serving.egg-info"
   rm -r ${BASE}/zoo/serving/bigdl_serving.egg-info
fi


serving_whl=dist/bigdl_serving-*.whl
serving_upload_command="twine upload ${serving_whl}"
echo "Command for uploading to pypi: $serving_upload_command, upload manually with this command"
