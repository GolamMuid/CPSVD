import sys
import subprocess

FILE_ID = "1rUOVCMcUWNoGNX_D6tliSzaMxpFulELn"
OUTPUT_FILE = "primevul.zip"


def install_gdown():
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "gdown"
    ])


try:
    import gdown
except ImportError:
    print("gdown not found. Installing...")
    install_gdown()
    import gdown


def download_file():
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    print("Starting download...")
    
    gdown.download(url, OUTPUT_FILE, quiet=False)
    
    print("Download complete:", OUTPUT_FILE)


if __name__ == "__main__":
    download_file()

# https://drive.google.com/file/d/1rUOVCMcUWNoGNX_D6tliSzaMxpFulELn/view?usp=sharing