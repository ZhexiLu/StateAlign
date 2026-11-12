#!/bin/bash
# Clone BFCL (the gorilla repository) at the pinned commit into third_party/gorilla and apply bfcl/bfcl.patch.
# The patch adds one flag to the command that starts the local vLLM server (--generation-config vllm), so that the
# sampling parameters BFCL does not set are vLLM's defaults and not those of the checkpoint's generation_config.json.
# BFCL's scorer, data and inference loop are not modified.
#   bash third_party/fetch.sh
# Then install BFCL into the evaluation environment:
#   pip install -e third_party/gorilla/berkeley-function-call-leaderboard
set -euo pipefail
cd "$(dirname "$0")"
URL=https://github.com/ShishirPatil/gorilla.git
COMMIT=6ea57973c7a6097fd7c5915698c54c17c5b1b6c8
if [ ! -d gorilla/.git ]; then git clone -q "$URL" gorilla; fi
git -C gorilla -c advice.detachedHead=false checkout -q "$COMMIT"
echo "gorilla at $(git -C gorilla rev-parse --short HEAD)"
if git -C gorilla apply --check ../bfcl/bfcl.patch 2>/dev/null; then
  git -C gorilla apply ../bfcl/bfcl.patch && echo "applied bfcl.patch"
else
  echo "bfcl.patch already applied (or the checkout is not clean)"
fi
