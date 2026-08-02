import subprocess
import sys
import importlib.metadata as metadata

REQUIRED_PACKAGES = {
    "numpy": "2.2.6",
    "torch": "2.10.0",
    "matplotlib": "3.10.8",
    "tqdm": "4.67.3",
    "scikit-learn": "1.7.2",
    "imbalanced-learn": "0.14.1",
    "xgboost": "3.2.0",
}


def get_installed_version(package_name):
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def install_package(package_name, version):
    target = f"{package_name}=={version}"
    print(f"Installing {target} ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--force-reinstall", "--no-deps", target
    ])


def main():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    for package, required_version in REQUIRED_PACKAGES.items():
        installed_version = get_installed_version(package)

        if installed_version == required_version:
            print(f"{package}: already at required version {required_version}, skipping.")
            continue

        if installed_version is None:
            print(f"{package}: not installed. Installing version {required_version}.")
        else:
            print(f"{package}: found version {installed_version}, "
                  f"forcing reinstall of {required_version}.")

        install_package(package, required_version)

    print("\nFinal installed versions:")
    for package in REQUIRED_PACKAGES:
        print(f"{package}: {get_installed_version(package)}")


if __name__ == "__main__":
    main()
