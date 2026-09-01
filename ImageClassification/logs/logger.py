"""
:file: logger.py
:date: 2026-08-20
:description: log analysis script for training logs
    read all subfolders in the current folder except for the backup folder.
    Each subfolder is named after the dataset, and the .log files are named as:
        {method}_{dataset_name}_{seed}.log     (note that dataset names like tiny_imagenet contain underscores)
    In each log file, it reads:
        - lines starting with ">>> Best Epoch: ", extracting Best Accuracy and Best Backdoor Accuracy
        - lines starting with ">>> Best time cost: ", extracting Best time cost
    Then it calculates the mean and std across different seeds, categorized by dataset and method, and prints the results to summary.md.
    Result format:
        - Each dataset has a title;
        - Each method has a line in the form of  mean (std);
        - Both Accuracies are multiplied by 100, keeping two decimal places, without the percent sign;
        - Time is kept to two decimal places.
    Scientific notation mode (--sci):
        - auto: default. Use scientific notation for four-digit numbers and above (absolute value >= 1000) or very small values (absolute value < 1e-4);
        - on  : force all numbers to use scientific notation;
        - off : force all numbers to use regular decimal format. 
"""

import os
import re
import argparse
import numpy as np

SEED_RE = re.compile(r"_(\d+)\.log$")

# ignore the backup folder, which may contain logs from previous runs
IGNORE_DIRS = {"backup"}

# scientific notation threshold
SCI_THRESHOLD = 1000 
SCI_SMALL_THRESHOLD = 1e-4


def parse_log_file(log_path): 
    best_acc = None
    best_backdoor = None
    best_time = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">>> Best Epoch: "):
                m = re.search(
                    r"Best Accuracy:\s*([0-9.eE+-]+),\s*Best Backdoor Accuracy:\s*([0-9.eE+-]+)",
                    line,
                )
                if m:
                    best_acc = float(m.group(1))
                    best_backdoor = float(m.group(2))
            elif line.startswith(">>> Best time cost: "):
                m = re.search(r"Best time cost:\s*([0-9.eE+-]+)", line)
                if m:
                    best_time = float(m.group(1))

    if best_acc is None or best_backdoor is None or best_time is None:
        return None

    return {"accuracy": best_acc, "backdoor": best_backdoor, "time": best_time}


def parse_filename(fname, dataset_name): 
    m = SEED_RE.search(fname)
    if not m:
        return None, None
    seed = int(m.group(1))
    stem = fname[: m.start()]
    suffix = "_{}".format(dataset_name)
    if stem.endswith(suffix):
        method = stem[: -len(suffix)]
    else:
        method = stem
    return method, seed


def fmt(val, sci_mode): 
    if sci_mode == "on":
        return "{:.2e}".format(val)
    if sci_mode == "off":
        return "{:.2f}".format(val)
    # auto
    absv = abs(val)
    if absv != 0 and (absv >= SCI_THRESHOLD or absv < SCI_SMALL_THRESHOLD):
        return "{:.2e}".format(val)
    return "{:.2f}".format(val)


def mean_std(sci_mode, *values): 
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return "N/A"
    values_np = np.array(vals)
    mean = float(values_np.mean())
    std = float(values_np.std()) if len(vals) > 1 else 0.0
    return "{} ({})".format(fmt(mean, sci_mode), fmt(std, sci_mode))


def main():
    parser = argparse.ArgumentParser(
        description="analyze training logs and generate summary.md",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sci",
        choices=["auto", "on", "off"],
        default="auto",
        help="scientific notation mode for numbers in the summary",
    )
    parser.add_argument(
        "--logs-dir",
        default=None,
        help="logs files directory, default is the current directory",
    )
    args = parser.parse_args()

    logs_dir = args.logs_dir or os.path.dirname(os.path.abspath(__file__))
    sci_mode = args.sci
    
    dataset_dirs = sorted(
        d
        for d in os.listdir(logs_dir)
        if os.path.isdir(os.path.join(logs_dir, d)) and d not in IGNORE_DIRS
    )

    # results structure: results[dataset][method] = {"accuracy": [...], "backdoor": [...], "time": [...]} 
    results = {}
    for dataset in dataset_dirs:
        dataset_path = os.path.join(logs_dir, dataset)
        results[dataset] = {}
        for fname in os.listdir(dataset_path):
            if not fname.endswith(".log"):
                continue
            method, seed = parse_filename(fname, dataset)
            if method is None:
                print("warning: failed to parse filename, skipping: {}".format(fname))
                continue
            log_path = os.path.join(dataset_path, fname)
            parsed = parse_log_file(log_path)
            if parsed is None:
                print("warning: missing key fields, skipping: {}".format(fname))
                continue
            entry = results[dataset].setdefault(
                method, {"accuracy": [], "backdoor": [], "time": []}
            )
            entry["accuracy"].append(parsed["accuracy"])
            entry["backdoor"].append(parsed["backdoor"])
            entry["time"].append(parsed["time"])

    
    lines = []
    lines.append("# Training Summary")
    lines.append("")
    for dataset in dataset_dirs:
        if not results[dataset]:
            continue
        lines.append("## {}".format(dataset))
        lines.append("")
        lines.append(
            "| Method | Accuracy | Backdoor Accuracy | Time (s) |"
        )
        lines.append(
            "|--------|----------|-------------------|----------|"
        )
        for method in sorted(results[dataset].keys()):
            entry = results[dataset][method]
            acc_str = mean_std(sci_mode, *[v * 100 for v in entry["accuracy"]])
            backdoor_str = mean_std(sci_mode, *[v * 100 for v in entry["backdoor"]])
            time_str = mean_std(sci_mode, *entry["time"])
            lines.append("| {} | {} | {} | {} |".format(method, acc_str, backdoor_str, time_str))
        lines.append("")

    summary_content = "\n".join(lines).rstrip() + "\n"

    summary_path = os.path.join(logs_dir, "temp_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_content)

    print("Generated summary: {}".format(summary_path))
    print(summary_content)


if __name__ == "__main__":
    main()
