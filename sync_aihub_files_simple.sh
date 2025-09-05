#!/bin/bash

# Simple AIHUB File Sync Script
# This script syncs files from NFS server based on paths in small_set.txt

set -e  # Exit on any error

# Configuration
SMALL_SET_FILE="data/replay_cl_250610/small_set.txt"
DRY_RUN=false  # Set to true for dry run
BATCH_SIZE=100  # Process files in batches

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

# Function to map paths
map_path() {
    local path="$1"
    
    if [[ "$path" =~ ^AIHUB_FreeCommunication_May/Bonafide/Training ]]; then
        local relative_path="${path#AIHUB_FreeCommunication_May/Bonafide/Training/}"
        echo "/datad/Datasets/AIHUB_SV/Training/$relative_path"
        echo "/nvme2/hungdx/Datasets/AIHUB_SV/Training/$relative_path"
    elif [[ "$path" =~ ^AIHUB_FreeCommunication_May/Bonafide/Validation ]]; then
        local relative_path="${path#AIHUB_FreeCommunication_May/Bonafide/Validation/}"
        echo "/datad/Datasets/AIHUB_SV/Validation/$relative_path"
        echo "/nvme2/hungdx/Datasets/AIHUB_SV/Validation/$relative_path"
    else
        echo ""
        echo ""
    fi
}

# Function to sync a single file
sync_file() {
    local nfs_path="$1"
    local local_path="$2"
    local dry_run="$3"
    
    if [[ "$dry_run" == "true" ]]; then
        print_info "[DRY RUN] Would sync: $nfs_path -> $local_path"
        return 0
    fi
    
    # Check if source exists
    if [[ ! -f "$nfs_path" ]]; then
        print_error "Source file does not exist: $nfs_path"
        return 1
    fi
    
    # Check if destination already exists
    if [[ -f "$local_path" ]]; then
        print_warning "File already exists, skipping: $local_path"
        return 0
    fi
    
    # Create destination directory
    local dest_dir=$(dirname "$local_path")
    mkdir -p "$dest_dir"
    
    # Copy file
    if cp "$nfs_path" "$local_path"; then
        print_info "SUCCESS: Synced $(basename "$nfs_path")"
        return 0
    else
        print_error "Failed to sync $nfs_path"
        return 1
    fi
}

# Main execution
main() {
    echo "Simple AIHUB File Sync Script"
    echo "=================================================="
    
    # Check if file exists
    if [[ ! -f "$SMALL_SET_FILE" ]]; then
        print_error "File not found: $SMALL_SET_FILE"
        exit 1
    fi
    
    # Check NFS accessibility
    print_info "Checking NFS accessibility..."
    if [[ ! -d "/datad/Datasets/AIHUB_SV/Training" ]]; then
        print_error "NFS Training directory not accessible"
        exit 1
    fi
    if [[ ! -d "/datad/Datasets/AIHUB_SV/Validation" ]]; then
        print_error "NFS Validation directory not accessible"
        exit 1
    fi
    print_info "NFS directories are accessible"
    
    # Extract file paths
    print_info "Extracting file paths..."
    local temp_file=$(mktemp)
    awk '{print $1}' "$SMALL_SET_FILE" > "$temp_file"
    local total_files=$(wc -l < "$temp_file")
    print_info "Found $total_files files to process"
    
    # Process files in batches
    local success_count=0
    local error_count=0
    local processed_count=0
    
    print_info "Starting batch processing (batch size: $BATCH_SIZE)..."
    
    while IFS= read -r original_path; do
        if [[ -z "$original_path" ]]; then
            continue
        fi
        
        ((processed_count++))
        
        # Map the path
        local paths=$(map_path "$original_path")
        local nfs_path=$(echo "$paths" | head -1)
        local local_path=$(echo "$paths" | tail -1)
        
        if [[ -z "$nfs_path" || -z "$local_path" ]]; then
            print_warning "Could not map path: $original_path"
            continue
        fi
        
        # Sync the file
        if sync_file "$nfs_path" "$local_path" "$DRY_RUN"; then
            ((success_count++))
        else
            ((error_count++))
        fi
        
        # Print progress
        if (( processed_count % 100 == 0 )); then
            print_info "Progress: $processed_count/$total_files files processed"
        fi
        
        # Process in batches to avoid hanging
        if (( processed_count % BATCH_SIZE == 0 )); then
            print_info "Completed batch $((processed_count / BATCH_SIZE)). Pausing for 1 second..."
            sleep 1
        fi
        
    done < "$temp_file"
    
    # Cleanup
    rm -f "$temp_file"
    
    # Print summary
    echo
    echo "=================================================="
    echo "SYNC SUMMARY"
    echo "Total files processed: $processed_count"
    echo "Successfully synced: $success_count"
    echo "Errors: $error_count"
    
    if [[ $error_count -gt 0 ]]; then
        print_warning "Some files failed to sync. Check the error messages above."
    elif [[ "$DRY_RUN" != "true" ]]; then
        print_info "All files synced successfully!"
        print_info "Files are now available at:"
        print_info "  /nvme2/hungdx/Datasets/AIHUB_SV/Training"
        print_info "  /nvme2/hungdx/Datasets/AIHUB_SV/Validation"
    fi
}

# Run main function
main "$@" 