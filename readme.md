## Installing packages

1. Run the install_packages.py file. It should install all the packages with exact version numbers.

`python3 install_packages.py`

2. In case the script fails to install all the packages, open the dependency.txt file. It contains all the dependencies and their version number. Install the package with thier version number manually.

## Downloading embedding

1. Go to embedding folder
2. Run download_primevul_emb.py file. It will download a zip file named as primevul.zip

`python3 download_primevul_emb.py`

3. Unzip the zip file.

## Downloading dataset (essential for scenario 5)

1. Go to dataset folder
2. Run download_primevul_raw.py file. It will download a zip file named as primevul.zip

`python3 download_primevul_raw.py`

3. Unzip the zip file.

## Running Scripts

1. Go to tests > synthetic oversampling
2. Run all the scripts from primevul_s1 > codet5 and codet5_dann
3. Run all the scripts from primevul_s2 > codet5 and codet5_dann
4. Run all the scripts from primevul_s3 > codebert_dann, codet5 and codet5_dann
5. Run all the scripts from primevul_s4 > codebert_dann, codet5 and codet5_dann
6. Run all the scripts from primevul_s5 > codet5
