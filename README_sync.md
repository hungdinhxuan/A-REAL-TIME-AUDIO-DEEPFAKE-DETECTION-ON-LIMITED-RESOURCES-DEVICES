# AIHUB File Sync Scripts

This directory contains scripts to sync AIHUB audio files from the NFS server based on the paths specified in `data/replay_cl_250610/small_set.txt`.

## Problem

The AIHUB dataset folders (`AIHUB_FreeCommunication_May/Bonafide/Training` and `AIHUB_FreeCommunication_May/Bonafide/Validation`) contain too many files and take a long time to sync completely from the NFS server. You only need the specific files listed in `small_set.txt`.

## Solution

These scripts will:
1. Read the file paths from `small_set.txt`
2. Map them to the correct NFS source paths and local destinations:
   - `AIHUB_FreeCommunication_May/Bonafide/Training` → `/datad/Datasets/AIHUB_SV/Training` → `/nvme2/hungdx/Datasets/AIHUB_SV/Training`
   - `AIHUB_FreeCommunication_May/Bonafide/Validation` → `/datad/Datasets/AIHUB_SV/Validation` → `/nvme2/hungdx/Datasets/AIHUB_SV/Validation`
3. Create necessary local directories
4. Sync only the required files to the clean directory structure

## Available Scripts

### 1. Python Script (`sync_aihub_files.py`)

**Features:**
- Pure Python implementation
- Detailed progress reporting
- Error handling and summary
- Dry run mode available

**Usage:**
```bash
# Make executable
chmod +x sync_aihub_files.py

# Run the script
./sync_aihub_files.py

# For dry run (see what would be synced without copying)
# Edit the script and set dry_run = True
```

### 2. Bash Script (`sync_aihub_files.sh`)

**Features:**
- Uses `rsync` for efficient file copying
- Colored output for better readability
- Faster execution for large files
- Built-in dry run mode

**Usage:**
```bash
# Make executable
chmod +x sync_aihub_files.sh

# Run the script
./sync_aihub_files.sh

# For dry run, edit the script and set DRY_RUN=true
```

## Configuration

Both scripts are configured to:
- Read from: `data/replay_cl_250610/small_set.txt`
- NFS Training path: `/datad/Datasets/AIHUB_SV/Training`
- NFS Validation path: `/datad/Datasets/AIHUB_SV/Validation`
- Local Training destination: `/nvme2/hungdx/Datasets/AIHUB_SV/Training`
- Local Validation destination: `/nvme2/hungdx/Datasets/AIHUB_SV/Validation`

## Example Output

```
AIHUB File Sync Script
==================================================
[INFO] Reading file paths from data/replay_cl_250610/small_set.txt...
[INFO] Found 5001 file paths
[INFO] Mapping to NFS source paths...
[INFO] Mapped 5001 files

Files by NFS source:
  /datad/Datasets/AIHUB_SV/Training: 4500 files
  /datad/Datasets/AIHUB_SV/Validation: 501 files

Files by local destination:
  /nvme2/hungdx/Datasets/AIHUB_SV/Training: 4500 files
  /nvme2/hungdx/Datasets/AIHUB_SV/Validation: 501 files

[INFO] Creating necessary directories...
[INFO] Created directory: /nvme2/hungdx/Datasets/AIHUB_SV/Training/data/TS_random_01/random/2021-12-15/0964
[INFO] Created directory: /nvme2/hungdx/Datasets/AIHUB_SV/Training/data/TS_random_02/random/2022-01-07/4040
...

[INFO] Starting file sync...
[INFO] Syncing: /datad/Datasets/AIHUB_SV/Training/data/TS_random_01/random/2021-12-15/0964/C0547-0964M1110-2__000_0-02074137.wav -> /nvme2/hungdx/Datasets/AIHUB_SV/Training/data/TS_random_01/random/2021-12-15/0964/C0547-0964M1110-2__000_0-02074137.wav
[INFO] SUCCESS: Synced C0547-0964M1110-2__000_0-02074137.wav
...

==================================================
SYNC SUMMARY
Total files processed: 5001
Successfully synced: 5001
Errors: 0

Files by destination:
  /nvme2/hungdx/Datasets/AIHUB_SV/Training: 4500 files
  /nvme2/hungdx/Datasets/AIHUB_SV/Validation: 501 files

[INFO] All files synced successfully!

Files are now available at:
  /nvme2/hungdx/Datasets/AIHUB_SV/Training
  /nvme2/hungdx/Datasets/AIHUB_SV/Validation
```

## Recommendations

1. **Use the bash script** (`sync_aihub_files.sh`) for better performance with large audio files
2. **Test with dry run first** to see what files would be synced
3. **Check available disk space** before running the full sync (ensure `/nvme2` has enough space)
4. **Monitor the NFS connection** during sync to ensure stability

## Troubleshooting

- **Permission errors**: Make sure you have read access to the NFS paths and write access to `/nvme2/hungdx/Datasets/`
- **Missing files**: Some files might not exist on the NFS server
- **Network issues**: Check NFS connectivity if sync fails
- **Disk space**: Ensure `/nvme2` has enough storage for all files
- **Directory creation errors**: Ensure you have write permissions to `/nvme2/hungdx/Datasets/`

## File Structure

After running the script, you'll have the following clean structure:
```
/nvme2/hungdx/Datasets/AIHUB_SV/
├── Training/
│   ├── data/
│   │   ├── TS_random_01/
│   │   ├── TS_random_02/
│   │   └── TS_common_01/
│   └── audio_3.studio_1/
└── Validation/
    └── data/
        ├── VS_random_01/
        └── VS_common_01/
```

## Benefits of New Structure

1. **Clean organization**: Files are stored in a dedicated dataset directory
2. **Easy access**: Direct paths without nested relative directories
3. **Better performance**: Local storage on fast NVMe drive
4. **Scalable**: Easy to add more datasets in the future
5. **Consistent**: Matches the original NFS structure but locally 