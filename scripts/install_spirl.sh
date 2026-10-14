#!/usr/bin/env bash
# Install the SPiRL reference code and its Python dependencies under $PAP_ROOT (default: current directory).
set -euo pipefail
cd "${PAP_ROOT:-.}"
mkdir -p external
if [[ ! -d external/spirl/.git ]]; then
  git clone https://github.com/clvrai/spirl.git external/spirl
fi
git -C external/spirl checkout e74ad23d2e9696369ec3d998f2920be4cddf9015
python -m pip install --target external/python_deps attrdict funcsigs future tensorboardX
