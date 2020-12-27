if [ ! -d "datasets" ]; then
  mkdir datasets
fi
cd datasets
if [ ! -f "cats_and_dogs_filtered.zip.tgz" ]; then
  wget https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip
  unzip cats_and_dogs_filtered.zip
fi

