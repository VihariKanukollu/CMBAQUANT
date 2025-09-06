#!/usr/bin/env python3
"""
Comprehensive WandB Run Analyzer v2.0
Analyzes training metrics and outputs detailed statistics in JSON format.
Enhanced for ACT/Quantum model analysis with advanced pattern detection.

Usage: 
1. Modify the RUN_PATH variable below with your wandb run path
2. Run: python3 wandb_analyzer.py

Output includes:
- Detailed statistical analysis of all metrics (min, max, mean, percentiles, trends, etc.)
- Training dynamics (runtime, learning rate schedules, convergence patterns)
- WandB metadata (system info, git commit, command arguments, environment details)
- Key performance indicators and improvement analysis
- Correlation analysis between critical metric pairs
- ACT-specific dynamics (halting patterns, Q-learning stability, efficiency metrics)
- Performance breakthrough detection and timing analysis
- MCP gate behavior analysis (cost trends, utilization, convergence)
- Training stability windows and coefficient of variation analysis
"""

import wandb
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, unquote

# =============================================================================
# CONFIGURATION - MODIFY THIS SECTION
# =============================================================================
# Change this to your wandb run path: "entity/project/run_id"
RUN_PATH = "viharikvs-urbankisaan/Sudoku-6x6-1000 ACT-torch/wpfc61u8"

# Output directory (relative to script location)
OUTPUT_DIR = "checkstop"
# =============================================================================

def calculate_metric_stats(series, metric_name):
    """Calculate comprehensive statistics for a metric series."""
    if series.empty or series.isna().all():
        return {
            "metric_name": metric_name,
            "available": False,
            "reason": "No data available"
        }
    
    # Remove NaN values for calculations
    clean_series = series.dropna()
    
    if len(clean_series) == 0:
        return {
            "metric_name": metric_name,
            "available": False,
            "reason": "All values are NaN"
        }
    
    stats = {
        "metric_name": metric_name,
        "available": True,
        "data_points": len(clean_series),
        "total_logged_points": len(series),
        "missing_values": len(series) - len(clean_series),
        
        # Basic statistics
        "min": float(clean_series.min()),
        "max": float(clean_series.max()),
        "mean": float(clean_series.mean()),
        "median": float(clean_series.median()),
        "std": float(clean_series.std()),
        "variance": float(clean_series.var()),
        
        # Percentiles
        "percentiles": {
            "p5": float(clean_series.quantile(0.05)),
            "p25": float(clean_series.quantile(0.25)),
            "p75": float(clean_series.quantile(0.75)),
            "p95": float(clean_series.quantile(0.95))
        },
        
        # Range analysis
        "range": float(clean_series.max() - clean_series.min()),
        "iqr": float(clean_series.quantile(0.75) - clean_series.quantile(0.25)),
        
        # Trend analysis
        "first_value": float(clean_series.iloc[0]),
        "last_value": float(clean_series.iloc[-1]),
        "absolute_change": float(clean_series.iloc[-1] - clean_series.iloc[0]),
        "percent_change": float((clean_series.iloc[-1] - clean_series.iloc[0]) / abs(clean_series.iloc[0]) * 100) if clean_series.iloc[0] != 0 else None,
        
        # Stability analysis
        "coefficient_of_variation": float(clean_series.std() / clean_series.mean()) if clean_series.mean() != 0 else None,
        
        # Moving averages (if enough data points)
        "moving_averages": {}
    }
    
    # Calculate moving averages if we have enough data
    if len(clean_series) >= 10:
        stats["moving_averages"]["last_10"] = float(clean_series.tail(10).mean())
    if len(clean_series) >= 50:
        stats["moving_averages"]["last_50"] = float(clean_series.tail(50).mean())
    if len(clean_series) >= 100:
        stats["moving_averages"]["last_100"] = float(clean_series.tail(100).mean())
    
    # Improvement analysis (for loss metrics - lower is better, for accuracy - higher is better)
    is_loss_metric = any(term in metric_name.lower() for term in ['loss', 'cost', 'error'])
    is_accuracy_metric = any(term in metric_name.lower() for term in ['accuracy', 'acc'])
    
    if is_loss_metric:
        stats["improvement_direction"] = "lower_is_better"
        stats["is_improving"] = stats["last_value"] < stats["first_value"]
        stats["best_value"] = stats["min"]
        stats["worst_value"] = stats["max"]
    elif is_accuracy_metric:
        stats["improvement_direction"] = "higher_is_better"
        stats["is_improving"] = stats["last_value"] > stats["first_value"]
        stats["best_value"] = stats["max"]
        stats["worst_value"] = stats["min"]
    else:
        stats["improvement_direction"] = "unknown"
        stats["is_improving"] = None
        stats["best_value"] = None
        stats["worst_value"] = None
    
    return stats

def analyze_value_frequencies(series, metric_name):
    """Count frequency of specific milestone values"""
    if series.empty:
        return {"available": False, "reason": "No data available"}
    
    # Define values of interest based on metric type
    if 'accuracy' in metric_name.lower():
        values_of_interest = [0.0, 0.5, 1.0]  # 0%, 50%, 100%
        milestone_names = ["zero_accuracy", "half_accuracy", "perfect_accuracy"]
    elif 'loss' in metric_name.lower():
        # For losses, look at very low values
        min_val = series.min()
        q25 = series.quantile(0.25)
        values_of_interest = [min_val, q25]
        milestone_names = ["minimum_loss", "low_quartile_loss"]
    else:
        # For other metrics, look at extremes
        values_of_interest = [series.min(), series.max()]
        milestone_names = ["minimum_value", "maximum_value"]
    
    frequency_analysis = {
        "available": True,
        "metric_name": metric_name,
        "total_measurements": len(series),
        "value_frequencies": {}
    }
    
    for val, name in zip(values_of_interest, milestone_names):
        if pd.notna(val):
            # Use tolerance for floating point comparison
            tolerance = 1e-6
            matches = (abs(series - val) < tolerance).sum()
            frequency_analysis["value_frequencies"][name] = {
                "target_value": float(val),
                "exact_count": int(matches),
                "percentage": float(matches / len(series) * 100),
                "first_occurrence": int(series[abs(series - val) < tolerance].index[0]) if matches > 0 else None,
                "last_occurrence": int(series[abs(series - val) < tolerance].index[-1]) if matches > 0 else None
            }
    
    # Additional analysis for accuracy metrics
    if 'accuracy' in metric_name.lower():
        # Count ranges
        ranges = [
            (0.0, 0.1, "very_low"),
            (0.1, 0.5, "low"),
            (0.5, 0.9, "high"), 
            (0.9, 1.0, "very_high")
        ]
        
        frequency_analysis["range_analysis"] = {}
        for low, high, name in ranges:
            in_range = ((series >= low) & (series < high)).sum()
            frequency_analysis["range_analysis"][name] = {
                "range": f"{low}-{high}",
                "count": int(in_range),
                "percentage": float(in_range / len(series) * 100)
            }
        
        # Perfect accuracy streak analysis
        perfect_mask = abs(series - 1.0) < 1e-6
        if perfect_mask.any():
            # Find consecutive streaks of perfect accuracy
            streaks = []
            current_streak = 0
            for val in perfect_mask:
                if val:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        streaks.append(current_streak)
                        current_streak = 0
            if current_streak > 0:
                streaks.append(current_streak)
            
            frequency_analysis["perfect_streaks"] = {
                "longest_streak": max(streaks) if streaks else 0,
                "total_streaks": len(streaks),
                "avg_streak_length": float(np.mean(streaks)) if streaks else 0
            }
    
    return frequency_analysis

def analyze_training_dynamics(history_df):
    """Analyze overall training dynamics and patterns."""
    dynamics = {
        "training_duration": {},
        "step_analysis": {},
        "convergence_analysis": {}
    }
    
    if '_runtime' in history_df.columns:
        runtime = history_df['_runtime'].dropna()
        if not runtime.empty:
            dynamics["training_duration"] = {
                "total_runtime_seconds": float(runtime.max()),
                "total_runtime_hours": float(runtime.max() / 3600),
                "avg_step_time": float(runtime.diff().mean()) if len(runtime) > 1 else None
            }
    
    if '_step' in history_df.columns:
        steps = history_df['_step'].dropna()
        if not steps.empty:
            dynamics["step_analysis"] = {
                "total_steps": int(steps.max() - steps.min()) if len(steps) > 1 else len(steps),
                "step_range": [int(steps.min()), int(steps.max())],
                "avg_step_increment": float(steps.diff().mean()) if len(steps) > 1 else None
            }
    
    # Learning rate analysis
    if 'train/lr' in history_df.columns:
        lr = history_df['train/lr'].dropna()
        if not lr.empty:
            dynamics["learning_rate"] = {
                "initial_lr": float(lr.iloc[0]),
                "final_lr": float(lr.iloc[-1]),
                "lr_schedule_detected": len(lr.unique()) > 1,
                "lr_decay_factor": float(lr.iloc[-1] / lr.iloc[0]) if lr.iloc[0] != 0 else None
            }
    
    return dynamics

def calculate_correlations(history_df):
    """Calculate correlations between key metric pairs."""
    correlations = {}
    
    # Key pairs for ACT models
    pairs = [
        ('train/steps', 'train/exact_accuracy'),
        ('train/steps', 'train/q_halt_accuracy'),
        ('train/mcp_cost', 'train/exact_accuracy'),
        ('train/q_halt_loss', 'train/q_continue_loss'),
        ('train/lm_loss', 'train/accuracy'),
        ('train/steps', 'train/lm_loss'),
        ('train/q_halt_accuracy', 'train/exact_accuracy'),
        ('train/mcp_cost', 'train/steps')
    ]
    
    for metric1, metric2 in pairs:
        if metric1 in history_df.columns and metric2 in history_df.columns:
            clean_data = history_df[[metric1, metric2]].dropna()
            if len(clean_data) > 2:
                correlation = clean_data.corr().iloc[0, 1]
                if not np.isnan(correlation):
                    correlations[f"{metric1}_vs_{metric2}"] = {
                        "correlation": float(correlation),
                        "strength": "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.4 else "weak",
                        "direction": "positive" if correlation > 0 else "negative",
                        "sample_size": len(clean_data)
                    }
    
    return correlations

def analyze_stability_windows(series, window_size=50, metric_name=""):
    """Identify stable vs unstable training periods."""
    if len(series) < window_size:
        return {
            "available": False,
            "reason": f"Insufficient data points ({len(series)} < {window_size})"
        }
    
    rolling_std = series.rolling(window=window_size).std()
    rolling_mean = series.rolling(window=window_size).mean()
    
    # Coefficient of variation for stability
    cv = rolling_std / rolling_mean.abs()
    cv = cv.replace([np.inf, -np.inf], np.nan).dropna()
    
    if cv.empty:
        return {"available": False, "reason": "No valid coefficient of variation data"}
    
    stability_threshold = cv.quantile(0.25)  # Bottom 25% = most stable
    
    stable_regions = []
    current_start = None
    
    for i, cv_val in enumerate(cv):
        if pd.notna(cv_val) and cv_val < stability_threshold:
            if current_start is None:
                current_start = i
        else:
            if current_start is not None:
                stable_regions.append((current_start, i))
                current_start = None
    
    # Close last region if still open
    if current_start is not None:
        stable_regions.append((current_start, len(cv)))
    
    total_stable_steps = sum(end - start for start, end in stable_regions)
    
    return {
        "available": True,
        "metric_name": metric_name,
        "stable_regions": stable_regions[:10],  # Top 10 regions
        "num_stable_regions": len(stable_regions),
        "total_stable_steps": total_stable_steps,
        "stability_ratio": float(total_stable_steps / len(series)) if len(series) > 0 else 0,
        "avg_cv": float(cv.mean()),
        "min_cv": float(cv.min()),
        "stability_threshold": float(stability_threshold)
    }

def analyze_act_dynamics(history_df):
    """Analyze ACT-specific patterns."""
    act_analysis = {}
    
    # Halting pattern analysis
    if 'train/steps' in history_df.columns:
        steps = history_df['train/steps'].dropna()
        if not steps.empty:
            step_counts = steps.value_counts()
            max_possible_steps = steps.max() if steps.max() > 0 else 100
            
            act_analysis["halting_patterns"] = {
                "uses_max_steps_ratio": float((steps == steps.max()).sum() / len(steps)),
                "uses_min_steps_ratio": float((steps == 0).sum() / len(steps)),
                "step_distribution": {
                    "unique_values": int(steps.nunique()),
                    "most_common_step": int(step_counts.index[0]) if not step_counts.empty else None,
                    "most_common_frequency": float(step_counts.iloc[0] / len(steps)) if not step_counts.empty else None,
                    "step_variance": float(steps.var()),
                    "step_entropy": float(-sum(p * np.log2(p) for p in step_counts / len(steps) if p > 0))
                },
                "efficiency_metrics": {
                    "avg_steps_used": float(steps.mean()),
                    "steps_utilization": float(steps.mean() / max_possible_steps) if max_possible_steps > 0 else None,
                    "halting_consistency": float(1.0 - steps.std() / steps.mean()) if steps.mean() > 0 else None
                }
            }
    
    # Q-learning convergence analysis  
    if 'train/q_halt_accuracy' in history_df.columns:
        q_acc = history_df['train/q_halt_accuracy'].dropna()
        if not q_acc.empty:
            binary_switches = (q_acc.diff().abs() > 0.5).sum()
            extreme_time = ((q_acc == 0) | (q_acc == 1)).sum()
            
            act_analysis["q_learning_stability"] = {
                "binary_switches": int(binary_switches),
                "switch_rate": float(binary_switches / len(q_acc)) if len(q_acc) > 0 else 0,
                "time_at_extremes": float(extreme_time / len(q_acc)) if len(q_acc) > 0 else 0,
                "convergence_quality": {
                    "final_value": float(q_acc.iloc[-1]),
                    "converged_to_one": float(q_acc.tail(50).mean()) > 0.8 if len(q_acc) >= 50 else None,
                    "learning_stability": float(q_acc.tail(100).std()) if len(q_acc) >= 100 else None
                }
            }
    
    # Continue vs Halt loss relationship
    if 'train/q_halt_loss' in history_df.columns and 'train/q_continue_loss' in history_df.columns:
        halt_loss = history_df['train/q_halt_loss'].dropna()
        continue_loss = history_df['train/q_continue_loss'].dropna()
        
        if not halt_loss.empty and not continue_loss.empty:
            # Align the series
            common_idx = halt_loss.index.intersection(continue_loss.index)
            if len(common_idx) > 10:
                aligned_halt = halt_loss.loc[common_idx]
                aligned_continue = continue_loss.loc[common_idx]
                
                act_analysis["halt_continue_dynamics"] = {
                    "loss_ratio_mean": float((aligned_halt / aligned_continue.replace(0, np.nan)).mean()),
                    "loss_difference_trend": "halt_decreasing" if aligned_halt.iloc[-1] < aligned_halt.iloc[0] else "halt_increasing",
                    "balanced_learning": abs(aligned_halt.mean() - aligned_continue.mean()) < 0.1
                }
    
    return act_analysis

def detect_breakthroughs(history_df, metrics=['train/exact_accuracy', 'train/accuracy'], threshold=0.3):
    """Detect when model achieves breakthrough performance."""
    all_breakthroughs = {}
    
    for metric in metrics:
        if metric not in history_df.columns:
            continue
            
        series = history_df[metric].dropna()
        if series.empty:
            continue
        
        breakthroughs = []
        for i in range(1, len(series)):
            improvement = series.iloc[i] - series.iloc[i-1]
            if improvement > threshold:
                breakthroughs.append({
                    "step": int(series.index[i]),
                    "improvement": float(improvement),
                    "from_value": float(series.iloc[i-1]),
                    "to_value": float(series.iloc[i]),
                    "relative_improvement": float(improvement / max(series.iloc[i-1], 0.001))
                })
        
        # Sort by improvement magnitude
        breakthroughs = sorted(breakthroughs, key=lambda x: x['improvement'], reverse=True)
        
        all_breakthroughs[metric] = {
            "breakthrough_count": len(breakthroughs),
            "major_breakthroughs": breakthroughs[:3],  # Top 3
            "first_success": int(series[series > 0.1].index[0]) if (series > 0.1).any() else None,
            "first_major_success": int(series[series > 0.5].index[0]) if (series > 0.5).any() else None,
            "breakthrough_timing": {
                "early_phase": len([b for b in breakthroughs if b['step'] < len(series) * 0.33]),
                "mid_phase": len([b for b in breakthroughs if len(series) * 0.33 <= b['step'] < len(series) * 0.66]),
                "late_phase": len([b for b in breakthroughs if b['step'] >= len(series) * 0.66])
            }
        }
    
    return all_breakthroughs

def analyze_mcp_gates(history_df):
    """Analyze MCP gate behavior over training."""
    if 'train/mcp_cost' not in history_df.columns:
        return {
            "available": False,
            "reason": "MCP cost not found - model may not be using MCP"
        }
    
    mcp_cost = history_df['train/mcp_cost'].dropna()
    if mcp_cost.empty:
        return {
            "available": False, 
            "reason": "No MCP cost data available"
        }
    
    # Trend analysis
    first_half = mcp_cost[:len(mcp_cost)//2]
    second_half = mcp_cost[len(mcp_cost)//2:]
    
    trend_direction = "decreasing" if mcp_cost.iloc[-1] < mcp_cost.iloc[0] else "increasing"
    trend_magnitude = abs(mcp_cost.iloc[-1] - mcp_cost.iloc[0]) / mcp_cost.iloc[0] if mcp_cost.iloc[0] != 0 else 0
    
    # Find convergence point (when cost stabilizes)
    rolling_std = mcp_cost.rolling(window=min(50, len(mcp_cost)//4)).std()
    stable_threshold = rolling_std.quantile(0.2)
    convergence_candidates = rolling_std[rolling_std < stable_threshold]
    
    analysis = {
        "available": True,
        "cost_dynamics": {
            "initial_cost": float(mcp_cost.iloc[0]),
            "final_cost": float(mcp_cost.iloc[-1]),
            "mean_cost": float(mcp_cost.mean()),
            "cost_range": float(mcp_cost.max() - mcp_cost.min()),
            "trend_direction": trend_direction,
            "trend_magnitude": float(trend_magnitude)
        },
        "learning_phases": {
            "early_mean": float(first_half.mean()),
            "late_mean": float(second_half.mean()),
            "phase_improvement": float((first_half.mean() - second_half.mean()) / first_half.mean()) if first_half.mean() != 0 else 0
        },
        "stability_analysis": {
            "coefficient_of_variation": float(mcp_cost.std() / mcp_cost.mean()) if mcp_cost.mean() != 0 else None,
            "convergence_point": int(convergence_candidates.index[0]) if not convergence_candidates.empty else None,
            "late_stability": float(mcp_cost.tail(100).std()) if len(mcp_cost) >= 100 else float(mcp_cost.tail(len(mcp_cost)//4).std()) if len(mcp_cost) >= 20 else None
        }
    }
    
    # Gate utilization analysis
    if mcp_cost.mean() > 0:
        analysis["gate_utilization"] = {
            "avg_gates_per_step": float(mcp_cost.mean()),
            "max_gate_usage": float(mcp_cost.max()),
            "gate_efficiency": "improving" if trend_direction == "decreasing" else "degrading",
            "utilization_consistency": float(1.0 - mcp_cost.std() / mcp_cost.mean())
        }
    
    return analysis

def generate_run_summary(run, history_df, metrics_analysis):
    """Generate a high-level summary of the run."""
    summary = {
        "run_info": {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "created_at": str(run.created_at),
            "project": run.project,
            "entity": run.entity
        },
        "model_info": {
            "num_parameters": run.summary.get('num_params', 'Unknown'),
            "model_architecture": run.config.get('arch', {}).get('name', 'Unknown')
        },
        "dataset_info": {
            "data_path": run.config.get('data_path', 'Unknown'),
            "global_batch_size": run.config.get('global_batch_size', 'Unknown')
        },
        "training_overview": {
            "total_metrics_tracked": len([m for m in metrics_analysis if m["available"]]),
            "total_data_points": len(history_df),
            "run_completed": run.state == "finished",
            "run_status": run.state
        }
    }
    
    # Key performance indicators
    kpis = {}
    
    # Find main accuracy metric
    accuracy_metrics = [m for m in metrics_analysis if 'accuracy' in m['metric_name'].lower() and m['available']]
    if accuracy_metrics:
        best_acc_metric = max(accuracy_metrics, key=lambda x: x.get('best_value', 0))
        kpis["best_accuracy"] = {
            "metric_name": best_acc_metric['metric_name'],
            "value": best_acc_metric['best_value'],
            "final_value": best_acc_metric['last_value']
        }
    
    # Find main loss metric
    loss_metrics = [m for m in metrics_analysis if 'loss' in m['metric_name'].lower() and m['available'] and 'halt' not in m['metric_name'].lower()]
    if loss_metrics:
        best_loss_metric = min(loss_metrics, key=lambda x: x.get('best_value', float('inf')))
        kpis["best_loss"] = {
            "metric_name": best_loss_metric['metric_name'],
            "value": best_loss_metric['best_value'],
            "final_value": best_loss_metric['last_value'],
            "improvement": best_loss_metric['absolute_change']
        }
    
    summary["key_performance_indicators"] = kpis
    
    return summary

def main():
    """Main analysis function."""
    print(f"Analyzing WandB run: {RUN_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    try:
        # Initialize WandB API
        api = wandb.Api()
        
        # Fetch the run
        print("Fetching run data...")
        def _normalize_run_path(p: str) -> str:
            """Accept either a full URL or 'entity/project/run_id'."""
            try:
                if p.startswith("http://") or p.startswith("https://"):
                    u = urlparse(p)
                    parts = [q for q in u.path.strip("/").split("/") if q]
                    # Expected: /{entity}/{project}/runs/{run_id}
                    if len(parts) >= 4 and parts[2] == "runs":
                        entity = parts[0]
                        project = unquote(parts[1])
                        run_id = parts[3]
                        return f"{entity}/{project}/{run_id}"
            except Exception:
                pass
            return p

        run = api.run(_normalize_run_path(RUN_PATH))
        
        # Get training history
        print("Downloading training history...")
        history_df = run.history()
        
        # Get wandb metadata and download checkpoints
        print("Fetching wandb metadata and checkpoints...")
        wandb_metadata = {}
        downloaded_files = {"checkpoints": [], "configs": []}
        
        try:
            # Create download directory for this run
            run_download_dir = os.path.join(OUTPUT_DIR, f"downloads_{run.id}")
            os.makedirs(run_download_dir, exist_ok=True)
            
            # Get all files from the run
            all_files = run.files()
            print(f"Found {len(all_files)} files in run")
            
            # Categorize and download files
            for file in all_files:
                try:
                    if file.name == "wandb-metadata.json":
                        # Download and read metadata
                        file.download(root=run_download_dir, replace=True)
                        metadata_path = os.path.join(run_download_dir, file.name)
                        with open(metadata_path, 'r') as f:
                            wandb_metadata = json.load(f)
                        print(f"  ✓ Downloaded metadata: {file.name}")
                        
                    elif file.name.endswith(('.pt', '.pth')) or 'step_' in file.name:
                        # Download checkpoint files
                        file.download(root=run_download_dir, replace=True)
                        checkpoint_path = os.path.join(run_download_dir, file.name)
                        downloaded_files["checkpoints"].append(checkpoint_path)
                        print(f"  ✓ Downloaded checkpoint: {file.name} ({file.size} bytes)")
                        
                    elif any(keyword in file.name.lower() for keyword in ['config', 'yaml']):
                        # Download config files
                        file.download(root=run_download_dir, replace=True)
                        config_path = os.path.join(run_download_dir, file.name)
                        downloaded_files["configs"].append(config_path)
                        print(f"  ✓ Downloaded config: {file.name}")
                        
                except Exception as e:
                    print(f"  ⚠ Failed to download {file.name}: {e}")
            
            print(f"\nDownload Summary:")
            print(f"  Checkpoints: {len(downloaded_files['checkpoints'])}")
            print(f"  Configs: {len(downloaded_files['configs'])}")
            print(f"  Metadata: {'✓' if wandb_metadata else '✗'}")
            
        except Exception as e:
            print(f"  Warning: Could not fetch run files: {e}")
        
        if history_df.empty:
            print("WARNING: No training history found!")
            return
        
        print(f"Found {len(history_df)} data points with {len(history_df.columns)} metrics")
        
        # Analyze each metric
        print("Analyzing metrics...")
        metrics_analysis = []
        
        for column in history_df.columns:
            if not column.startswith('_'):  # Skip internal wandb columns
                print(f"  Analyzing {column}...")
                metric_stats = calculate_metric_stats(history_df[column], column)
                
                # Add value frequency analysis for key metrics
                if any(key_term in column.lower() for key_term in ['accuracy', 'loss', 'steps']):
                    metric_stats["value_frequencies"] = analyze_value_frequencies(history_df[column].dropna(), column)
                
                metrics_analysis.append(metric_stats)
        
        # Analyze training dynamics
        print("Analyzing training dynamics...")
        training_dynamics = analyze_training_dynamics(history_df)
        
        # Generate run summary
        print("Generating run summary...")
        run_summary = generate_run_summary(run, history_df, metrics_analysis)
        
        # Advanced ACT/Quantum analysis
        print("Analyzing correlations...")
        correlations = calculate_correlations(history_df)
        
        print("Analyzing ACT dynamics...")
        act_dynamics = analyze_act_dynamics(history_df)
        
        print("Detecting performance breakthroughs...")
        breakthroughs = detect_breakthroughs(history_df)
        
        print("Analyzing MCP gates...")
        mcp_analysis = analyze_mcp_gates(history_df)
        
        # Stability analysis for key metrics
        print("Analyzing training stability...")
        stability_analysis = {}
        key_metrics = ['train/lm_loss', 'train/exact_accuracy', 'train/steps', 'train/mcp_cost']
        for metric in key_metrics:
            if metric in history_df.columns:
                stability_analysis[metric] = analyze_stability_windows(
                    history_df[metric].dropna(), 
                    window_size=min(50, len(history_df)//4),
                    metric_name=metric
                )
        
        # Compile final analysis
        final_analysis = {
            "analysis_metadata": {
                "generated_at": datetime.now().isoformat(),
                "wandb_run_path": RUN_PATH,
                "analyzer_version": "2.2"
            },
            "run_summary": run_summary,
            "training_dynamics": training_dynamics,
            "metrics_analysis": metrics_analysis,
            "correlation_analysis": correlations,
            "act_dynamics": act_dynamics,
            "performance_breakthroughs": breakthroughs,
            "mcp_gate_analysis": mcp_analysis,
            "stability_analysis": stability_analysis,
            "downloaded_files": downloaded_files,
            "wandb_metadata": wandb_metadata,
            "raw_config": dict(run.config),
            "raw_summary": dict(run.summary)
        }
        
        # Save to JSON file
        output_file = os.path.join(OUTPUT_DIR, f"analysis_{run.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(final_analysis, f, indent=2, default=str)
        
        print(f"\n{'='*60}")
        print(f"ANALYSIS COMPLETE!")
        print(f"{'='*60}")
        print(f"Output file: {output_file}")
        print(f"Run: {run.name} ({run.id})")
        print(f"State: {run.state}")
        print(f"Metrics analyzed: {len(metrics_analysis)}")
        print(f"Data points: {len(history_df)}")
        
        # Quick insights
        available_metrics = [m for m in metrics_analysis if m["available"]]
        print(f"\nQuick Insights:")
        print(f"  - Available metrics: {len(available_metrics)}/{len(metrics_analysis)}")
        print(f"  - Wandb metadata: {'✓ Captured' if wandb_metadata else '✗ Not found'}")
        print(f"  - Downloaded checkpoints: {len(downloaded_files['checkpoints'])}")
        print(f"  - Downloaded configs: {len(downloaded_files['configs'])}")
        
        # Show checkpoint info if available
        if downloaded_files["checkpoints"]:
            print(f"  - Checkpoint files ready for evaluation:")
            for checkpoint in downloaded_files["checkpoints"]:
                print(f"    {os.path.basename(checkpoint)}")
        
        if wandb_metadata:
            if 'os' in wandb_metadata:
                print(f"  - OS: {wandb_metadata['os']}")
            if 'python' in wandb_metadata:
                print(f"  - Python: {wandb_metadata['python']}")
            if 'git' in wandb_metadata and 'commit' in wandb_metadata['git']:
                commit = wandb_metadata['git']['commit'][:8]
                print(f"  - Git commit: {commit}")
        
        if run_summary.get("key_performance_indicators", {}).get("best_accuracy"):
            acc_info = run_summary["key_performance_indicators"]["best_accuracy"]
            print(f"  - Best accuracy: {acc_info['value']:.4f} ({acc_info['metric_name']})")
        
        if run_summary.get("key_performance_indicators", {}).get("best_loss"):
            loss_info = run_summary["key_performance_indicators"]["best_loss"]
            print(f"  - Best loss: {loss_info['value']:.4f} ({loss_info['metric_name']})")
            if loss_info['improvement']:
                print(f"  - Loss improvement: {loss_info['improvement']:.4f}")
        
        # Show perfect accuracy counts if available
        for metric in metrics_analysis:
            if metric['available'] and 'exact_accuracy' in metric['metric_name'] and 'value_frequencies' in metric:
                freq_data = metric['value_frequencies']
                if freq_data['available'] and 'perfect_accuracy' in freq_data['value_frequencies']:
                    perfect_count = freq_data['value_frequencies']['perfect_accuracy']['exact_count']
                    perfect_pct = freq_data['value_frequencies']['perfect_accuracy']['percentage']
                    print(f"  - Perfect accuracy achieved: {perfect_count} times ({perfect_pct:.1f}%)")
                    
                    if 'perfect_streaks' in freq_data and freq_data['perfect_streaks']['longest_streak'] > 0:
                        longest = freq_data['perfect_streaks']['longest_streak']
                        total_streaks = freq_data['perfect_streaks']['total_streaks']
                        print(f"  - Longest perfect streak: {longest} steps ({total_streaks} streaks total)")
        
        return final_analysis
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print("\nPlease check:")
        print("1. You're logged into wandb (python3 -m wandb whoami)")
        print("2. The RUN_PATH is correct")
        print("3. You have access to the project")
        return None

if __name__ == "__main__":
    result = main()
    if result:
        print("\nAnalysis saved successfully!")
    else:
        print("\nAnalysis failed!")
