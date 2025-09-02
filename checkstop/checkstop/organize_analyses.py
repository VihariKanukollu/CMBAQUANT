#!/usr/bin/env python3
"""
Organize WandB analysis JSON files by task domain
"""

import json
import os
import shutil
from pathlib import Path

def get_task_domain(analysis_file):
    """Determine task domain from analysis JSON"""
    try:
        with open(analysis_file, 'r') as f:
            data = json.load(f)
        
        # Check wandb_run_path or project info
        run_path = data.get('analysis_metadata', {}).get('wandb_run_path', '')
        project = data.get('run_summary', {}).get('run_info', {}).get('project', '')
        
        # Determine domain from project/path name
        full_context = (run_path + ' ' + project).lower()
        
        if 'sudoku-4x4' in full_context:
            return 'sudoku_4x4'
        elif 'sudoku-6x6' in full_context:
            return 'sudoku_6x6'  
        elif 'sudoku-extreme' in full_context or 'sudoku' in full_context:
            return 'sudoku_9x9'  # Default for extreme/regular sudoku
        elif 'arc' in full_context:
            return 'arc'
        elif 'maze' in full_context:
            return 'maze'
        else:
            return 'unknown'
            
    except Exception as e:
        print(f"Error reading {analysis_file}: {e}")
        return 'unknown'

def main():
    """Organize all analysis files by task domain"""
    
    print("Organizing WandB analysis files by task domain...")
    
    # Find all JSON analysis files
    json_files = [f for f in os.listdir('.') if f.startswith('analysis_') and f.endswith('.json')]
    
    print(f"Found {len(json_files)} analysis files to organize")
    
    # Create domain directories
    domains = ['sudoku_4x4', 'sudoku_6x6', 'sudoku_9x9', 'arc', 'maze', 'unknown']
    for domain in domains:
        os.makedirs(domain, exist_ok=True)
    
    # Organize files
    organized_count = {domain: 0 for domain in domains}
    
    for json_file in json_files:
        domain = get_task_domain(json_file)
        
        # Copy to domain folder
        src = json_file
        dst = os.path.join(domain, json_file)
        
        try:
            shutil.copy2(src, dst)
            organized_count[domain] += 1
            print(f"  {json_file} → {domain}/")
        except Exception as e:
            print(f"  ❌ Failed to move {json_file}: {e}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"ORGANIZATION SUMMARY")
    print(f"{'='*50}")
    
    for domain, count in organized_count.items():
        if count > 0:
            print(f"📁 {domain}: {count} files")
    
    print(f"\n🎯 Total organized: {sum(organized_count.values())} files")
    print(f"📂 Domain folders created with analysis files")

if __name__ == "__main__":
    main()
