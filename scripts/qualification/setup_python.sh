#!/usr/bin/env bash
set -euo pipefail
qualification_root=${TYR_QUALIFICATION_ROOT:-"$HOME/tyr-qualification"}
bootstrap_python=${TYR_QUALIFICATION_BOOTSTRAP_PYTHON:-/home/pehle/dev/tyr/.venv-gpu/bin/python}
mkdir -p "$qualification_root"
if [[ ! -x "$qualification_root/venv/bin/python" ]]; then
  "$bootstrap_python" -m venv "$qualification_root/venv"
fi
# Reuse the runner's immutable CUDA Torch wheel without altering its environment.
# Pinned reference dependencies installed below take precedence in the new venv.
torch_site=$("$bootstrap_python" -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).resolve().parent.parent)')
purelib=$("$qualification_root/venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
printf '%s\n' "$torch_site" > "$purelib/spark-existing-runtime.pth"
"$qualification_root/venv/bin/python" -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  -r scripts/qualification/requirements.txt
