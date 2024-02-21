import os
import shutil
import argparse
from pathlib import Path

def update_html_in_parent_folder(folder_path):
    # Get parent folder
    parent_folder = Path(folder_path).parent

    # List all html files under parent folder and delete them
    for html_file in parent_folder.glob('*.html'):
        html_file.unlink()

    # Find latest html file under folder_path
    latest_html_file = max(Path(folder_path).glob('*.html'), key=os.path.getctime, default=None)

    # Copy the latest html file to parent folder
    if latest_html_file is not None:
        shutil.copy(latest_html_file, parent_folder)
    
    print(latest_html_file.name)

def main():
    parser = argparse.ArgumentParser(description="Update HTML in parent folder.")
    parser.add_argument("-f", "--folder", type=str, help="Path to the folder")
    args = parser.parse_args()

    update_html_in_parent_folder(args.folder)

if __name__ == "__main__":
    main()
