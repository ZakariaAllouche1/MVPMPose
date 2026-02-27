import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "script1.py",
    "script2.py",
    "script3.py",
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