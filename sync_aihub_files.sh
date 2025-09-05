#!/bin/bash

# AIHUB File Sync Script using rsync
# This script syncs files from NFS server based on paths in small_set.txt

set -e  # Exit on any error

# Configuration
SMALL_SET_FILE="data/replay_cl_250610/small_set.txt"
DRY_RUN=false  # Set to true for dry run

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Function to extract file paths from small_set.txt
extract_paths() {
    local file="$1"
    if [[ ! -f "$file" ]]; then
        print_error "File not found: $file"
        exit 1
    fi
    
    # Extract first column (file paths) and remove duplicates
    awk '{print $1}' "$file" | sort -u
}

# Function to map paths to NFS sources and local destinations
map_to_nfs_and_local() {
    local path="$1"
    
    if [[ "$path" =~ ^AIHUB_FreeCommunication_May/Bonafide/Training ]]; then
        echo "/datad/Datasets/AIHUB_SV/Training"
    elif [[ "$path" =~ ^AIHUB_FreeCommunication_May/Bonafide/Validation ]]; then
        echo "/datad/Datasets/AIHUB_SV/Validation"
    else
        echo ""
    fi
}

# Function to get local destination path
get_local_dest() {
    local path="$1"
    
    if [[ "$path" =~ ^AIHUB_FreeCommunication_May/Bonafide/Training ]]; then
        echo "/nvme2/hungdx/Datasets/AIHUB_SV/Training"
    elif [[ "$path" =~ ^AIHUB_FreeCommunication_May/Bonafide/Validation ]]; then
        echo "/nvme2/hungdx/Datasets/AIHUB_SV/Validation"
    else
        echo ""
    fi
}

# Function to get relative path
get_relative_path() {
    local path="$1"
    
    if [[ "$path" =~ ^AIHUB_FreeCommunication_May/Bonafide/Training(.*)$ ]]; then
        echo "${BASH_REMATCH[1]}"
    elif [[ "$path" =~ ^AIHUB_FreeCommunication_May/Bonafide/Validation(.*)$ ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo ""
    fi
}

# Function to create directories
create_directories() {
    local paths_file="$1"
    
    print_info "Creating necessary directories..."
    
    while IFS= read -r path; do
        if [[ -n "$path" ]]; then
            local nfs_source=$(map_to_nfs_and_local "$path")
            local local_dest=$(get_local_dest "$path")
            local relative_path=$(get_relative_path "$path")
            
            if [[ -n "$nfs_source" && -n "$local_dest" && -n "$relative_path" ]]; then
                local full_local_path="$local_dest/$relative_path"
                local dir=$(dirname "$full_local_path")
                if [[ ! -d "$dir" ]]; then
                    mkdir -p "$dir"
                    print_info "Created directory: $dir"
                fi
            fi
        fi
    done < "$paths_file"
}

# Function to sync files using rsync
sync_files() {
    local paths_file="$1"
    local dry_run="$2"
    
    local success_count=0
    local error_count=0
    local total_count=0
    local processed_count=0
    
    # Track files by destination for summary
    declare -A dest_counts
    dest_counts["/nvme2/hungdx/Datasets/AIHUB_SV/Training"]=0
    dest_counts["/nvme2/hungdx/Datasets/AIHUB_SV/Validation"]=0
    
    print_info "Starting file sync..."
    
    if [[ "$dry_run" == "true" ]]; then
        print_warning "DRY RUN MODE - No files will be copied"
    fi
    
    # Count total files first
    local total_files=$(wc -l < "$paths_file")
    print_info "Total files to process: $total_files"
    
    # Test first few files
    print_info "Testing first 5 files..."
    local test_count=0
    while IFS= read -r original_path && (( test_count < 5 )); do
        if [[ -z "$original_path" ]]; then
            continue
        fi
        
        ((test_count++))
        print_info "Test file $test_count: $original_path"
        
        local nfs_source=$(map_to_nfs_and_local "$original_path")
        local local_dest=$(get_local_dest "$original_path")
        local relative_path=$(get_relative_path "$original_path")
        
        print_debug "  NFS source: $nfs_source"
        print_debug "  Local dest: $local_dest"
        print_debug "  Relative path: $relative_path"
        
        if [[ -z "$nfs_source" || -z "$local_dest" || -z "$relative_path" ]]; then
            print_warning "Could not map path: $original_path"
            continue
        fi
        
        local nfs_path="$nfs_source/$relative_path"
        local local_path="$local_dest/$relative_path"
        
        print_debug "  Full NFS path: $nfs_path"
        print_debug "  Full local path: $local_path"
        
        # Check if source exists
        if [[ ! -f "$nfs_path" ]]; then
            print_error "Source file does not exist: $nfs_path"
            continue
        fi
        
        print_info "  Source file exists: $nfs_path"
        
    done < "$paths_file"
    
    print_info "Test completed. Starting full sync..."
    
    # Reset file pointer
    while IFS= read -r original_path; do
        if [[ -z "$original_path" ]]; then
            continue
        fi
        
        ((processed_count++))
        print_debug "Processing file $processed_count/$total_files: $original_path"
        
        local nfs_source=$(map_to_nfs_and_local "$original_path")
        local local_dest=$(get_local_dest "$original_path")
        local relative_path=$(get_relative_path "$original_path")
        
        if [[ -z "$nfs_source" || -z "$local_dest" || -z "$relative_path" ]]; then
            print_warning "Could not map path: $original_path"
            continue
        fi
        
        local nfs_path="$nfs_source/$relative_path"
        local local_path="$local_dest/$relative_path"
        ((total_count++))
        ((dest_counts["$local_dest"]++))
        
        print_info "Syncing: $nfs_path -> $local_path"
        
        # Check if source exists
        if [[ ! -f "$nfs_path" ]]; then
            print_error "Source file does not exist: $nfs_path"
            ((error_count++))
            continue
        fi
        
        # Check if destination already exists
        if [[ -f "$local_path" ]]; then
            print_warning "File already exists, skipping: $local_path"
            ((success_count++))
            continue
        fi
        
        # Create destination directory if it doesn't exist
        local dest_dir=$(dirname "$local_path")
        mkdir -p "$dest_dir"
        
        # Use rsync for efficient copying
        local rsync_opts="-av"
        if [[ "$dry_run" == "true" ]]; then
            rsync_opts="$rsync_opts --dry-run"
        fi
        
        print_debug "Running: rsync $rsync_opts \"$nfs_path\" \"$local_path\""
        
        if rsync $rsync_opts "$nfs_path" "$local_path" >/dev/null 2>&1; then
            print_info "SUCCESS: Synced $(basename "$nfs_path")"
            ((success_count++))
        else
            print_error "Failed to sync $nfs_path"
            ((error_count++))
        fi
        
        # Print progress every 100 files
        if (( processed_count % 100 == 0 )); then
            print_info "Progress: $processed_count/$total_files files processed"
        fi
        
    done < "$paths_file"
    
    echo
    echo "=================================================="
    echo "SYNC SUMMARY"
    echo "Total files processed: $total_count"
    echo "Successfully synced: $success_count"
    echo "Errors: $error_count"
    echo
    echo "Files by destination:"
    for dest in "${!dest_counts[@]}"; do
        echo "  $dest: ${dest_counts[$dest]} files"
    done
    
    if [[ $error_count -gt 0 ]]; then
        print_warning "Some files failed to sync. Check the error messages above."
    elif [[ "$dry_run" != "true" ]]; then
        print_info "All files synced successfully!"
        echo
        print_info "Files are now available at:"
        for dest in "${!dest_counts[@]}"; do
            echo "  $dest"
        done
    fi
}

# Main execution
main() {
    echo "AIHUB File Sync Script"
    echo "=================================================="
    
    # Check if small_set.txt exists
    if [[ ! -f "$SMALL_SET_FILE" ]]; then
        print_error "File not found: $SMALL_SET_FILE"
        exit 1
    fi
    
    # Check if NFS is accessible
    print_info "Checking NFS accessibility..."
    if [[ ! -d "/datad/Datasets/AIHUB_SV/Training" ]]; then
        print_error "NFS Training directory not accessible: /datad/Datasets/AIHUB_SV/Training"
        exit 1
    fi
    if [[ ! -d "/datad/Datasets/AIHUB_SV/Validation" ]]; then
        print_error "NFS Validation directory not accessible: /datad/Datasets/AIHUB_SV/Validation"
        exit 1
    fi
    print_info "NFS directories are accessible"
    
    # Check local destination directories
    print_info "Checking local destination directories..."
    if [[ ! -d "/nvme2/hungdx/Datasets" ]]; then
        print_error "Local datasets directory not accessible: /nvme2/hungdx/Datasets"
        exit 1
    fi
    print_info "Local directories are accessible"
    
    # Extract unique file paths
    print_info "Reading file paths from $SMALL_SET_FILE..."
    local temp_paths_file=$(mktemp)
    extract_paths "$SMALL_SET_FILE" > "$temp_paths_file"
    local path_count=$(wc -l < "$temp_paths_file")
    print_info "Found $path_count unique file paths"
    
    # Create directories
    create_directories "$temp_paths_file"
    
    # Sync files
    sync_files "$temp_paths_file" "$DRY_RUN"
    
    # Cleanup
    rm -f "$temp_paths_file"
}

# Run main function
main "$@" 