
FOLDER=elasticsearch-7.15.0
if [ -d "$FOLDER" ]; then
    echo "$FOLDER exists. Starting Elasticsearch."
    ./elasticsearch-7.15.0/bin/elasticsearch
else
    echo; echo "$FOLDER file not found!"
    echo "Please run setup.sh script to install requirements."; echo
    exit 1
fi