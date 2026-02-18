"""Install all dependencies for the GLP misalignment experiment."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "requirements.txt"


def run(cmd):
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result


if __name__ == "__main__":
    print("Installing requirements...")
    run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    print("\nAll dependencies installed successfully.")
    print("\nNote: The GLP package is installed from GitHub:")
    print("  https://github.com/g-luo/generative_latent_prior")
    print("\nIf the GLP install fails, install it manually with:")
    print("  pip install git+https://github.com/g-luo/generative_latent_prior.git")
