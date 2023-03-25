wget -c https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz
tar -xvf imagenetv2-matched-frequency.tar.gz
mv imagenetv2-matched-frequency-format-val val2
rm ImageNetV2-matched-frequency.tar.gz
python rename_files.py ./val2
