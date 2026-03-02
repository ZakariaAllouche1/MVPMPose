import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "Codeocclusion.py",
    "Codeocclusion_V2.py",
    "Codeocclusion_V3.py",
    "Codeocclusion_V2_detect.py",
]

def run(script_name: str) -> int:
    script_path = Path(__file__).with_name(script_name)
    if not script_path.exists():
        print(f"ERROR: File not found: {script_path}")
        return 2

    print(f"\nRunning: {script_path.name}")
    completed = subprocess.run([sys.executable, str(script_path)])
    return completed.returncode

def main():
    failures = []

    for s in SCRIPTS:
        code = run(s)
        if code != 0:
            print(f"WARNING: {s} failed (exit code {code}). Continuing.")
            failures.append((s, code))

    print("\nSummary")
    if failures:
        for s, code in failures:
            print(f"FAILED: {s} (exit code {code})")
        sys.exit(1)
    else:
        print("All scripts completed successfully.")
        sys.exit(0)

if __name__ == "__main__":
    main()