"""
Script for comparing configurations and identifying possible causes of metric degradation.
"""

import os
import json
from pathlib import Path
import mmengine
from deepdiff import DeepDiff


def load_config(config_path):
    """Loads configuration from file."""
    if config_path.endswith('.py'):
        cfg = mmengine.Config.fromfile(config_path)
        return dict(cfg)
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported configuration format: {config_path}")


def compare_configs(config1_path, config2_path, output_path=None):
    """
    Compares two configurations and outputs differences.
    
    Args:
        config1_path: Path to first configuration (original)
        config2_path: Path to second configuration (current)
        output_path: Path to save report (optional)
    """
    print(f"Loading configurations...")
    print(f"  Original: {config1_path}")
    print(f"  Current: {config2_path}")
    
    config1 = load_config(config1_path)
    config2 = load_config(config2_path)
    
    # Convert to dict if needed
    if hasattr(config1, '__dict__'):
        config1 = dict(config1)
    if hasattr(config2, '__dict__'):
        config2 = dict(config2)
    
    # Compare
    diff = DeepDiff(config1, config2, ignore_order=True, verbose_level=2)
    
    print("\n" + "="*80)
    print("CONFIGURATION DIFFERENCES")
    print("="*80)
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("CONFIGURATION DIFFERENCES")
    report_lines.append("="*80)
    report_lines.append(f"\nOriginal: {config1_path}")
    report_lines.append(f"Current: {config2_path}\n")
    
    if not diff:
        print("Configurations are identical!")
        report_lines.append("Configurations are identical!")
    else:
        # Critical differences (may affect metrics)
        critical_keys = [
            'dataset_eval_basedir',
            'mapping_iters', 'refine_map_iter',
            'num_iter',
            'semantic_device',
            'enable_active_planning',
            'max_exploration_steps',
            'eval_during_training',
            'eval_during_training_freq',
        ]
        
        print("\n🔴 CRITICAL DIFFERENCES (may affect metrics):")
        report_lines.append("\n🔴 CRITICAL DIFFERENCES (may affect metrics):")
        
        critical_found = False
        for key in critical_keys:
            val1 = get_nested_value(config1, key)
            val2 = get_nested_value(config2, key)
            if val1 != val2:
                critical_found = True
                print(f"  {key}:")
                print(f"    Original: {val1}")
                print(f"    Current:  {val2}")
                report_lines.append(f"  {key}:")
                report_lines.append(f"    Original: {val1}")
                report_lines.append(f"    Current:  {val2}")
        
        if not critical_found:
            print("  No critical differences found")
            report_lines.append("  No critical differences found")
        
        # All differences
        print("\n📋 ALL DIFFERENCES:")
        report_lines.append("\n📋 ALL DIFFERENCES:")
        
        if 'values_changed' in diff:
            print("\nChanged values:")
            report_lines.append("\nChanged values:")
            for key, change in diff['values_changed'].items():
                print(f"  {key}:")
                print(f"    Was: {change['old_value']}")
                print(f"    Now: {change['new_value']}")
                report_lines.append(f"  {key}:")
                report_lines.append(f"    Was: {change['old_value']}")
                report_lines.append(f"    Now: {change['new_value']}")
        
        if 'dictionary_item_added' in diff:
            print("\nAdded parameters:")
            report_lines.append("\nAdded parameters:")
            for item in diff['dictionary_item_added']:
                print(f"  {item}")
                report_lines.append(f"  {item}")
        
        if 'dictionary_item_removed' in diff:
            print("\nRemoved parameters:")
            report_lines.append("\nRemoved parameters:")
            for item in diff['dictionary_item_removed']:
                print(f"  {item}")
                report_lines.append(f"  {item}")
    
    # Save report
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"\nReport saved to {output_path}")
    
    return diff


def get_nested_value(d, key):
    """Gets value from nested dictionary by key (supports nesting via dot notation)."""
    keys = key.split('.')
    value = d
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return None
    return value


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare configurations to identify causes of metric degradation')
    parser.add_argument('--original', type=str, 
                       default='ActiveSGM-Orig/configs/Replica/office0/ActiveSem.py',
                       help='Path to original configuration')
    parser.add_argument('--current', type=str,
                       default='ActiveSGM/configs/Replica/office0/ActiveSem.py',
                       help='Path to current configuration')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save report (optional)')
    
    args = parser.parse_args()
    
    # Check file existence
    if not os.path.exists(args.original):
        print(f"Error: file {args.original} not found!")
        return
    
    if not os.path.exists(args.current):
        print(f"Error: file {args.current} not found!")
        return
    
    compare_configs(args.original, args.current, args.output)


if __name__ == "__main__":
    main()
