## Installing packages

1. Run the install_packages.py file. It should install all the packages with exact version numbers.

`python3 install_packages.py`

2. In case the script fails to install all the packages, open the dependency.txt file. It contains all the dependencies and their version number. Install the package with thier version number manually.

## pytorch on cuda

1. Run the check_and_fix_cuda_pytorch.py file. It will check if the pytorch for cuda is installed or not and will install if needed.

   `python3 check_and_fix_cuda_pytorch.py`

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

1. Go to tests > scripts
2. Run all the scripts from primevul_s1 > codetbert_dann
3. Run all the scripts from primevul_s2 > codetbert_dann
4. Run all the scripts from primevul_s3 > codetbert_dann
5. Run all the scripts from primevul_s4 > codetbert_dann
