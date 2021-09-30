
FILE=elasticsearch-7.15.0-linux-x86_64.tar.gz
if [ -f "$FILE" ]; then
    echo "$FILE exists. Type './run.sh' to start Elasticsearch."
else
    echo "Installation file '$FILE' does not exist. Downloading from 'https://artifacts.elastic.co/downloads/elasticsearch', and installing."
    # Download and install archive for Linux
    wget https://artifacts.elastic.co/downloads/elasticsearch/$FILE
    wget https://artifacts.elastic.co/downloads/elasticsearch/$FILE.sha512
    # Compares the SHA of the downloaded .tar.gz archive and the published checksum,
    shasum -a 512 -c $FILE.sha512
    # Extract tar file
    tar -xzf $FILE
    echo "Installation of Elasticsearch is completed. Please type './run.sh' to start service."
fi