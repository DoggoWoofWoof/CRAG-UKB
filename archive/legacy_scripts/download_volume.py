import modal
import os

def download_all(vol, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    
    print("Fetching list of all files...")
    entries = []
    for entry in vol.iterdir("/", recursive=True):
        if entry.type != modal.volume.FileEntryType.DIRECTORY:
            entries.append(entry)
            
    print(f"Found {len(entries)} files. Starting download...")
    
    for entry in entries:
        local_path = os.path.join(local_dir, entry.path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Simple resume: if file exists and size matches, skip
        if hasattr(entry, 'size') and os.path.exists(local_path) and os.path.getsize(local_path) == entry.size:
            print(f"Skipping {entry.path} (already downloaded)")
            continue
            
        size_str = f" ({entry.size} bytes)" if hasattr(entry, 'size') else ""
        print(f"Downloading {entry.path}{size_str}...")
        
        # Download file
        with open(local_path, "wb") as f:
            for chunk in vol.read_file(entry.path):
                f.write(chunk)

if __name__ == "__main__":
    print("Starting volume download...")
    vol = modal.Volume.from_name("crag-data-volume")
    download_all(vol, "crag_data_backup")
    print("Done!")
