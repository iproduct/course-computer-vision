import os
import urllib.request


def download_stitching_demo_images():
    # 1. Define the base URL pointing to OpenCV's official testing data
    base_url = "https://githubusercontent.com"

    # 2. Select a classic overlapping pair (boat1 and boat2)
    files_to_download = {
        "boat1.png": base_url + "boat1.png",
        "boat2.png": base_url + "boat2.png"
    }

    # 3. Create a target directory locally if it doesn't exist
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing download from opencv/opencv_extra repository...")

    # 4. Loop and safely retrieve the binary image streams
    for filename, url in files_to_download.items():
        destination_path = os.path.join(output_dir, filename)

        # Check if the file was already downloaded to prevent redundant bandwidth use
        if os.path.exists(destination_path):
            print(f" -> {filename} already exists in '{output_dir}/'. Skipping download.")
            continue

        print(f" -> Downloading {filename} from remote server...")
        try:
            # Execute standard HTTP request
            with urllib.request.urlopen(url, timeout=15) as response:
                with open(destination_path, "wb") as local_file:
                    local_file.write(response.read())
            print(f"Successfully saved: {destination_path}")
        except Exception as e:
            print(f"ERROR: Could not download {filename}. Reason: {e}")

    print("\nDownload process complete. Workspace is ready for feature matching!")


if __name__ == "__main__":
    download_stitching_demo_images()
