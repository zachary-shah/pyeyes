import tarfile
import urllib.request
from pathlib import Path

# Test data as of pyeyes v0.4.0
TEST_DATASET_URL = "https://github.com/zachary-shah/pyeyes/releases/download/test-data-v0.4.0/test-data-v0.4.0.tar.gz"


def download_test_data():
    """Download test dataset (takes 1-2 mins.)."""

    test_data_path = Path(__file__).parent / "test-data"

    # If data already exists, skip download
    if test_data_path.exists() and any(test_data_path.iterdir()):
        print("Test data already exists!")
        return test_data_path

    # Download and extract test data
    print(f"Downloading test data from {TEST_DATASET_URL}...")

    test_data_path.parent.mkdir(parents=True, exist_ok=True)
    tar_path = test_data_path.parent / "test-data.tar.gz"

    try:
        urllib.request.urlretrieve(TEST_DATASET_URL, tar_path)
        print("Extracting test data...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(test_data_path.parent)
        tar_path.unlink()  # Clean up
        print("Test data ready!")
    except Exception as e:
        raise RuntimeError(f"Failed to download test data: {e}")

    # remove hidden files
    hidden_file = test_data_path.parent / "._test-data"
    if hidden_file.exists():
        hidden_file.unlink()
    for file in test_data_path.glob("**/*"):
        if file.is_file() and file.name.startswith("._"):
            file.unlink()

    print(f"Test data ready at {test_data_path}")


if __name__ == "__main__":
    download_test_data()
