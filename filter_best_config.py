#!/usr/bin/env python3
"""
Script to filter experiment configuration with highest eval_accuracy from Ray Tune results.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple


def load_experiment_state(json_file_path: str) -> Dict[str, Any]:
    """Load the experiment state JSON file."""
    try:
        with open(json_file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {json_file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file {json_file_path}: {e}")
        sys.exit(1)


def extract_trial_data(experiment_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract trial data from experiment state."""
    trials = []
    
    # The experiment state contains checkpoints which are JSON strings
    for checkpoint_str in experiment_state.get('checkpoints', []):
        try:
            trial_data = json.loads(checkpoint_str)
            
            # Extract key information
            trial_info = {
                'trial_id': trial_data.get('trial_id'),
                'config': trial_data.get('config', {}),
                'status': trial_data.get('status'),
                'eval_accuracy': None,
                'last_result': trial_data.get('_last_result', {})
            }
            
            # Get eval_accuracy from _last_result if available
            if 'eval_accuracy' in trial_info['last_result']:
                trial_info['eval_accuracy'] = trial_info['last_result']['eval_accuracy']
            
            trials.append(trial_info)
            
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse checkpoint data: {e}")
            continue
    
    return trials


def find_best_trial(trials: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find the trial with the highest eval_accuracy."""
    valid_trials = [trial for trial in trials if trial['eval_accuracy'] is not None]
    
    if not valid_trials:
        print("No trials with eval_accuracy found.")
        return None
    
    best_trial = max(valid_trials, key=lambda x: x['eval_accuracy'])
    return best_trial


def print_trial_summary(trial: Dict[str, Any]):
    """Print a summary of the trial."""
    print(f"Best Trial ID: {trial['trial_id']}")
    print(f"Eval Accuracy: {trial['eval_accuracy']:.6f}")
    print(f"Status: {trial['status']}")
    print("\nConfiguration:")
    for key, value in trial['config'].items():
        print(f"  {key}: {value}")


def save_best_config(trial: Dict[str, Any], output_file: str):
    """Save the best configuration to a file."""
    config_data = {
        'trial_id': trial['trial_id'],
        'eval_accuracy': trial['eval_accuracy'],
        'status': trial['status'],
        'config': trial['config']
    }
    
    with open(output_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"\nBest configuration saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter experiment configuration with highest eval_accuracy from Ray Tune results"
    )
    parser.add_argument(
        "experiment_state_file",
        help="Path to the Ray Tune experiment state JSON file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file to save the best configuration (optional)"
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all trials with their eval_accuracy values"
    )
    
    args = parser.parse_args()
    
    # Load experiment state
    print(f"Loading experiment state from: {args.experiment_state_file}")
    experiment_state = load_experiment_state(args.experiment_state_file)
    
    # Extract trial data
    trials = extract_trial_data(experiment_state)
    print(f"Found {len(trials)} trials")
    
    # Show all trials if requested
    if args.show_all:
        print("\nAll trials with eval_accuracy:")
        valid_trials = [trial for trial in trials if trial['eval_accuracy'] is not None]
        valid_trials.sort(key=lambda x: x['eval_accuracy'], reverse=True)
        
        for i, trial in enumerate(valid_trials, 1):
            print(f"{i}. Trial {trial['trial_id']}: eval_accuracy = {trial['eval_accuracy']:.6f}")
    
    # Find best trial
    best_trial = find_best_trial(trials)
    
    if best_trial is None:
        print("No valid trials found.")
        return
    
    # Print results
    print("\n" + "="*60)
    print("BEST CONFIGURATION")
    print("="*60)
    print_trial_summary(best_trial)
    
    # Save to file if requested
    if args.output:
        save_best_config(best_trial, args.output)


if __name__ == "__main__":
    main()
