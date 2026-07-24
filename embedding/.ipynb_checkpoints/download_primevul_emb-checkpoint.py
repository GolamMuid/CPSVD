import sys
import subprocess

FILE_ID = "10ig55O-IQBUEpr_nMbgjSLhzrYw2iaCn"
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

# https://drive.google.com/file/d/10ig55O-IQBUEpr_nMbgjSLhzrYw2iaCn/view?usp=sharing