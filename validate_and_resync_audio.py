#!/usr/bin/env python3
"""
Script to validate audio files and re-sync corrupted ones
Uses librosa and ffmpeg to check audio file integrity
"""

import os
import shutil
import subprocess
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import librosa
import soundfile as sf

def check_audio_with_librosa(file_path):
    """Check if audio file can be loaded with librosa"""
    try:
        # Try to load audio file
        y, sr = librosa.load(file_path, sr=None, duration=1.0)  # Load only first second for speed
        if len(y) == 0:
            return False, "Empty audio"
        return True, "OK"
    except Exception as e:
        return False, str(e)

def check_audio_with_ffmpeg(file_path):
    """Check if audio file is valid using ffmpeg"""
    try:
        # Run ffmpeg to validate the file
        result = subprocess.run([
            'ffmpeg', '-v', 'error', '-i', file_path, 
            '-f', 'null', '-'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and not result.stderr:
            return True, "OK"
        else:
            return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)

def check_audio_with_soundfile(file_path):
    """Check if audio file can be read with soundfile"""
    try:
        with sf.SoundFile(file_path) as f:
            # Try to read a small chunk
            frames = f.read(1024, dtype='float32')
            if len(frames) == 0:
                return False, "Empty audio"
        return True, "OK"
    except Exception as e:
        return False, str(e)

def validate_audio_file(file_path, method='all'):
    """Validate audio file using specified method(s)"""
    results = {}
    
    if method in ['all', 'librosa']:
        results['librosa'] = check_audio_with_librosa(file_path)
    
    if method in ['all', 'soundfile']:
        results['soundfile'] = check_audio_with_soundfile(file_path)
    
    if method in ['all', 'ffmpeg']:
        results['ffmpeg'] = check_audio_with_ffmpeg(file_path)
    
    # File is valid if at least one method succeeds
    valid = any(result[0] for result in results.values())
    
    # Collect all error messages
    errors = [f"{method}: {result[1]}" for method, result in results.items() if not result[0]]
    error_msg = "; ".join(errors) if errors else "OK"
    
    return valid, error_msg, results

def find_audio_files(directory):
    """Find all audio files in directory"""
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
    audio_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in audio_extensions:
                audio_files.append(os.path.join(root, file))
    
    return audio_files

def map_local_to_nfs(local_path):
    """Map local file path back to NFS path"""
    # Convert local path to NFS path
    if '/nvme2/hungdx/Datasets/AIHUB_SV/Training/data' in local_path:
        nfs_path = local_path.replace('/nvme2/hungdx/Datasets/AIHUB_SV/Training/data', 
                                    '/datad/Datasets/AIHUB_SV/Training/data')
        return nfs_path
    elif '/nvme2/hungdx/Datasets/AIHUB_SV/Validation/data' in local_path:
        nfs_path = local_path.replace('/nvme2/hungdx/Datasets/AIHUB_SV/Validation/data', 
                                    '/datad/Datasets/AIHUB_SV/Validation/data')
        return nfs_path
    return None

def resync_file(nfs_path, local_path):
    """Re-sync a single file from NFS to local"""
    try:
        if not os.path.exists(nfs_path):
            return False, f"Source file does not exist: {nfs_path}"
        
        # Create directory if needed
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Copy file
        shutil.copy2(nfs_path, local_path)
        return True, "Re-synced successfully"
    except Exception as e:
        return False, str(e)

def main():
    # Configuration
    local_directories = [
        "/nvme2/hungdx/Datasets/AIHUB_SV/Training/data",
        "/nvme2/hungdx/Datasets/AIHUB_SV/Validation/data"
    ]
    
    validation_method = 'soundfile'  # 'librosa', 'soundfile', 'ffmpeg', or 'all'
    remove_corrupted = True  # Set to False to just report without removing
    resync_corrupted = True  # Set to False to just remove without re-syncing
    
    print("🎵 Audio File Validation and Re-sync Tool")
    print("=" * 60)
    print(f"🔍 Validation method: {validation_method}")
    print(f"🗑️  Remove corrupted: {remove_corrupted}")
    print(f"🔄 Re-sync corrupted: {resync_corrupted}")
    print()
    
    all_corrupted = []
    all_valid = []
    
    for directory in local_directories:
        if not os.path.exists(directory):
            print(f"❌ Directory not found: {directory}")
            continue
            
        print(f"📁 Processing directory: {directory}")
        
        # Find all audio files
        print("🔍 Finding audio files...")
        audio_files = find_audio_files(directory)
        print(f"✅ Found {len(audio_files)} audio files")
        
        if len(audio_files) == 0:
            continue
        
        # Validate audio files
        print("🔍 Validating audio files...")
        corrupted_files = []
        valid_files = []
        
        for file_path in tqdm(audio_files, desc="Validating", unit="file"):
            is_valid, error_msg, details = validate_audio_file(file_path, validation_method)
            
            if is_valid:
                valid_files.append(file_path)
            else:
                corrupted_files.append((file_path, error_msg))
                tqdm.write(f"❌ CORRUPTED: {os.path.basename(file_path)} - {error_msg}")
        
        print(f"\n📊 Results for {directory}:")
        print(f"✅ Valid files: {len(valid_files)}")
        print(f"❌ Corrupted files: {len(corrupted_files)}")
        
        all_valid.extend(valid_files)
        all_corrupted.extend(corrupted_files)
        
        # Remove corrupted files
        if remove_corrupted and corrupted_files:
            print(f"\n🗑️  Removing {len(corrupted_files)} corrupted files...")
            for file_path, error_msg in tqdm(corrupted_files, desc="Removing", unit="file"):
                try:
                    os.remove(file_path)
                    tqdm.write(f"🗑️  Removed: {os.path.basename(file_path)}")
                except Exception as e:
                    tqdm.write(f"❌ Failed to remove {file_path}: {e}")
        
        # Re-sync corrupted files
        if resync_corrupted and corrupted_files:
            print(f"\n🔄 Re-syncing {len(corrupted_files)} corrupted files...")
            resync_success = 0
            resync_failed = 0
            
            for file_path, error_msg in tqdm(corrupted_files, desc="Re-syncing", unit="file"):
                nfs_path = map_local_to_nfs(file_path)
                if nfs_path:
                    success, msg = resync_file(nfs_path, file_path)
                    if success:
                        resync_success += 1
                        # Validate the re-synced file
                        is_valid, new_error, _ = validate_audio_file(file_path, validation_method)
                        if is_valid:
                            tqdm.write(f"✅ Re-synced and validated: {os.path.basename(file_path)}")
                        else:
                            tqdm.write(f"⚠️  Re-synced but still corrupted: {os.path.basename(file_path)} - {new_error}")
                    else:
                        resync_failed += 1
                        tqdm.write(f"❌ Failed to re-sync {os.path.basename(file_path)}: {msg}")
                else:
                    resync_failed += 1
                    tqdm.write(f"❌ Could not map to NFS path: {file_path}")
            
            print(f"\n📊 Re-sync results:")
            print(f"✅ Successfully re-synced: {resync_success}")
            print(f"❌ Failed to re-sync: {resync_failed}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("📈 FINAL SUMMARY")
    print(f"Total audio files processed: {len(all_valid) + len(all_corrupted)}")
    print(f"✅ Valid files: {len(all_valid)}")
    print(f"❌ Corrupted files: {len(all_corrupted)}")
    
    if all_corrupted:
        print(f"\n📝 Corrupted files list:")
        for file_path, error_msg in all_corrupted[:10]:  # Show first 10
            print(f"  ❌ {os.path.basename(file_path)}: {error_msg}")
        if len(all_corrupted) > 10:
            print(f"  ... and {len(all_corrupted) - 10} more files")
    
    if len(all_corrupted) == 0:
        print("🎉 All audio files are valid!")
    else:
        print(f"⚠️  Found {len(all_corrupted)} corrupted files")

if __name__ == "__main__":
    main()