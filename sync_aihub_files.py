#!/usr/bin/env python3
"""
Script to sync AIHUB files from NFS server based on paths in small_set.txt
"""

import os
import shutil
import subprocess
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def read_file_paths(file_path):
    """Read file paths from the small_set.txt file"""
    paths = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Extract the first column (file path)
                parts = line.split()
                if parts:
                    paths.append(parts[0])
    return paths

def map_to_nfs_paths(file_paths):
    """Map relative paths to NFS source paths and local destination paths"""
    nfs_mapping = {
        'AIHUB_FreeCommunication_May/Bonafide/Training/data': {
            'nfs_source': '/datad/Datasets/AIHUB_SV/Training/data',
            'local_dest': '/nvme2/hungdx/Datasets/AIHUB_SV/Training/data'
        },
        'AIHUB_FreeCommunication_May/Bonafide/Validation/data': {
            'nfs_source': '/datad/Datasets/AIHUB_SV/Validation/data',
            'local_dest': '/nvme2/hungdx/Datasets/AIHUB_SV/Validation/data'
        }
    }
    
    mapped_files = []
    for path in file_paths:
        # Find the matching NFS prefix
        nfs_source = None
        local_dest = None
        relative_path = None
        
        for prefix, mapping in nfs_mapping.items():
            if path.startswith(prefix):
                nfs_source = mapping['nfs_source']
                local_dest = mapping['local_dest']
                relative_path = path[len(prefix):].lstrip('/')
                break
        
        if nfs_source and local_dest and relative_path:
            mapped_files.append({
                'original_path': path,
                'nfs_source': nfs_source,
                'local_dest': local_dest,
                'relative_path': relative_path,
                'full_nfs_path': os.path.join(nfs_source, relative_path),
                'full_local_path': os.path.join(local_dest, relative_path)
            })
        else:
            print(f"Warning: Could not map path {path}")
    
    return mapped_files

def create_directories(mapped_files):
    """Create necessary directories for the files"""
    dirs_to_create = set()
    for file_info in mapped_files:
        local_dir = os.path.dirname(file_info['full_local_path'])
        dirs_to_create.add(local_dir)
    
    print(f"Creating {len(dirs_to_create)} directories...")
    for dir_path in tqdm(sorted(dirs_to_create), desc="Creating directories", unit="dir"):
        os.makedirs(dir_path, exist_ok=True)

def sync_files(mapped_files, dry_run=False):
    """Sync files from NFS to local paths with progress tracking"""
    success_count = 0
    error_count = 0
    skip_count = 0
    
    print(f"\nSyncing {len(mapped_files)} files...")
    
    # Use tqdm for progress tracking
    for file_info in tqdm(mapped_files, desc="Syncing files", unit="file"):
        local_path = file_info['full_local_path']
        nfs_path = file_info['full_nfs_path']
        
        if dry_run:
            tqdm.write(f"[DRY RUN] Would sync: {os.path.basename(nfs_path)}")
            success_count += 1
            continue
        
        try:
            # Check if source file exists
            if not os.path.exists(nfs_path):
                tqdm.write(f"ERROR: Source file does not exist: {nfs_path}")
                error_count += 1
                continue
            
            # Check if destination already exists
            if os.path.exists(local_path):
                tqdm.write(f"SKIP: File already exists: {os.path.basename(local_path)}")
                skip_count += 1
                continue
            
            # Copy the file
            shutil.copy2(nfs_path, local_path)
            success_count += 1
            
        except Exception as e:
            tqdm.write(f"ERROR: Failed to sync {nfs_path}: {e}")
            error_count += 1
    
    return success_count, error_count, skip_count

def main():
    # Configuration
    small_set_file = "data/replay_cl_250610/small_set.txt"
    dry_run = False  # Set to True to see what would be synced without actually copying
    
    print("AIHUB File Sync Script with Progress Tracking")
    print("=" * 60)
    
    # Check if small_set.txt exists
    if not os.path.exists(small_set_file):
        print(f"❌ Error: {small_set_file} not found!")
        return
    
    # Check if NFS directories are accessible
    nfs_training = "/datad/Datasets/AIHUB_SV/Training"
    nfs_validation = "/datad/Datasets/AIHUB_SV/Validation"
    local_base = "/nvme2/hungdx/Datasets/AIHUB_SV"
    
    print("🔍 Checking accessibility...")
    if not os.path.exists(nfs_training):
        print(f"❌ Error: NFS Training directory not accessible: {nfs_training}")
        return
    if not os.path.exists(nfs_validation):
        print(f"❌ Error: NFS Validation directory not accessible: {nfs_validation}")
        return
    if not os.path.exists(os.path.dirname(local_base)):
        print(f"❌ Error: Local destination directory not accessible: {os.path.dirname(local_base)}")
        return
    
    print("✅ All directories are accessible")
    
    # Read file paths
    print(f"\n📖 Reading file paths from {small_set_file}...")
    try:
        file_paths = read_file_paths(small_set_file)
        print(f"✅ Found {len(file_paths)} file paths")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    # Map to NFS paths
    print("\n🗺️  Mapping to NFS source paths...")
    mapped_files = map_to_nfs_paths(file_paths)
    print(f"✅ Mapped {len(mapped_files)} files")
    
    if len(mapped_files) == 0:
        print("❌ No files could be mapped. Check the file format.")
        return
    
    # Group by NFS source for summary
    nfs_sources = defaultdict(list)
    for file_info in mapped_files:
        nfs_sources[file_info['nfs_source']].append(file_info)
    
    print("\n📊 Files by NFS source:")
    for nfs_source, files in nfs_sources.items():
        print(f"  📁 {nfs_source}: {len(files)} files")
    
    # Group by local destination for summary
    local_dests = defaultdict(list)
    for file_info in mapped_files:
        local_dests[file_info['local_dest']].append(file_info)
    
    print("\n📊 Files by local destination:")
    for local_dest, files in local_dests.items():
        print(f"  💾 {local_dest}: {len(files)} files")
    
    # Create directories
    print("\n📁 Creating necessary directories...")
    try:
        create_directories(mapped_files)
        print("✅ Directories created successfully")
    except Exception as e:
        print(f"❌ Error creating directories: {e}")
        return
    
    # Sync files
    if dry_run:
        print("\n🔍 DRY RUN MODE - No files will be copied")
    
    try:
        success_count, error_count, skip_count = sync_files(mapped_files, dry_run)
    except Exception as e:
        print(f"❌ Error during sync: {e}")
        return
    
    print("\n" + "=" * 60)
    print("📈 SYNC SUMMARY")
    print(f"Total files processed: {len(mapped_files)}")
    print(f"✅ Successfully synced: {success_count}")
    print(f"⏭️  Skipped (already exist): {skip_count}")
    print(f"❌ Errors: {error_count}")
    
    if error_count > 0:
        print("\n⚠️  Some files failed to sync. Check the error messages above.")
    elif not dry_run:
        print("\n🎉 All files synced successfully!")
        print(f"\n📂 Files are now available at:")
        for local_dest in local_dests.keys():
            print(f"  💾 {local_dest}")
    else:
        print("\n✅ Dry run completed successfully!")

if __name__ == "__main__":
    main() 