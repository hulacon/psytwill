#!/usr/bin/env bash
# Build the shared "stimfeat" conda environment on Talapas.
#
# One env for the stimulus-feature family — viz2psy + aud2psy + word2psy +
# psytwill, editable-installed side by side on a single torch/CUDA build.
# Per constellation-contracts.md §10 (mmmdata-agents repo): Python 3.12 is
# the constellation version standard; one solve, one numpy/torch/
# transformers, tested together; the lock manifest written here IS the
# version standard for the four repos. psytwill carries this script and the
# lock because it is the family's aggregation surface; the env serves all
# four repos equally.
#
# Adapted from duckbrain's scripts/setup_env.sh:
#   - channel handling: the FSL installer writes ~/.condarc with
#     `channels: #!final`, which CONDARC/CONDA_CHANNELS/`conda env create`
#     cannot beat; only `--override-channels -c conda-forge` on a plain
#     `conda create` works (checked on-cluster 2026-08-04, duckbrain).
#   - PYTHONNOUSERSITE=1 guard: ~/.local site-packages sit AHEAD of a conda
#     env's own site-packages (venvs are immune, conda envs are not).
#   - shared prefix + group rwX/setgid + `.conda-prefix` marker per repo.
# Deliberate difference: the payload installs via pip, not conda-forge —
# torch cu128 wheels, GitHub-only deepgaze-pytorch, and most of the
# extractor stack (resmem, open-clip-torch, faster-whisper, fasttext-wheel,
# ultralytics, mosqito, ...) are pip-only. Conda supplies python/pip/ffmpeg.
#
#   ./scripts/setup_env.sh                # build at the shared prefix
#   ./scripts/setup_env.sh --prefix PATH  # elsewhere (personal build)
#
# Cold pip cache downloads ~10 GB of wheels; run somewhere with network
# (a login node is fine — the work is network-bound, not CPU-bound).
# Rebuild policy: single writer. Delete the prefix and rebuild fresh before
# committing a new lock (an incremental solve is not a fresh one).
set -euo pipefail

umask 0002  # group-writable files from birth; parent setgid dirs handle group

CODE=/gpfs/projects/hulacon/shared/mmmdata/code
ENVS=/projects/hulacon/shared/envs
SHARED_PREFIX=$ENVS/stimfeat
PREFIX=$SHARED_PREFIX
CACHE=$ENVS/cache
CONDA_MODULE=miniconda3/20260319
REPOS=(viz2psy aud2psy word2psy psytwill)
OWNER=$CODE/psytwill              # where the lock manifest is committed
# cu128 matches the A100 nodes and the retired personal viz2psy env
# (torch 2.10.0+cu128, known-good on Talapas GPUs).
TORCH_INDEX=https://download.pytorch.org/whl/cu128
# Not on PyPI; commits pinned = the builds the family has been running on.
# deepgaze-pytorch imports OpenAI's `clip` (also GitHub-only) and `einops`
# at package import since its CLIP/DINOv2 feature extractors landed —
# undeclared in its metadata, so listed here explicitly.
DEEPGAZE='deepgaze-pytorch @ git+https://github.com/matthias-k/DeepGaze.git@7698f8344d84f9755dd913be58d554ea8936d926'
OPENAI_CLIP='clip @ git+https://github.com/openai/CLIP.git@ded190a052fdf4585bd685cee5bc96e0310d2c93'
# The retired personal viz2psy venv doubled as the workhorse env for
# mmmdata's converter/triage scripts (edf_triage, behavioral_convert,
# generate_inventory, validate_step8 — see mmmdata-docs/bidsification.md).
# Those scripts now point here, so carry their deps too.
COMPANIONS=(eyelinkio pybids nibabel pydicom)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)  PREFIX="${2:?--prefix needs a path}"; shift 2 ;;
    -h|--help) sed -n '2,35p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# conda: a bare `conda` on $PATH here is FSL's — load the module explicitly.
source /etc/profile.d/z00_lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module load "$CONDA_MODULE" 2>/dev/null || {
  echo "error: could not 'module load $CONDA_MODULE'" >&2; exit 1
}

export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="$CACHE/pip"   # shared cache; also spares /home quota

# Shared model caches — created here, exported for this build (so anything
# that downloads during verification lands shared), and exported for every
# user by the activate.d hook written below.
mkdir -p "$CACHE"/{pip,huggingface,torch,word2psy,nltk_data,ultralytics} "$ENVS/bin"
export HF_HOME="$CACHE/huggingface"
# Credentials stay private: HF_HOME is group-readable, so its default
# token file would be too. Read the token from the user's home instead.
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export TORCH_HOME="$CACHE/torch"
export WORD2PSY_CACHE="$CACHE/word2psy"
export NLTK_DATA="$CACHE/nltk_data"
export YOLO_CONFIG_DIR="$CACHE/ultralytics"

echo "prefix : $PREFIX"
echo "conda  : python=3.12 pip ffmpeg (conda-forge, --override-channels)"
echo "pip    : ${REPOS[*]} (editable, [dev]) + deepgaze-pytorch @ pinned git"
echo "torch  : $TORCH_INDEX"
echo

if [[ -d "$PREFIX" && -n "$(ls -A "$PREFIX" 2>/dev/null)" ]]; then
  echo "==> prefix exists; conda step skipped (delete it for a fresh solve)"
else
  echo "==> conda create"
  conda create --prefix "$PREFIX" --override-channels -c conda-forge --yes \
    python=3.12 pip ffmpeg
fi

PY="$PREFIX/bin/python"

echo
echo "==> pip install (one resolve: all four repos + deepgaze; ~10 GB cold)"
"$PY" -m pip install --extra-index-url "$TORCH_INDEX" \
  "$DEEPGAZE" "$OPENAI_CLIP" einops "${COMPANIONS[@]}" \
  -e "$CODE/viz2psy[dev,interactive]" \
  -e "$CODE/aud2psy[dev]" \
  -e "$CODE/word2psy[dev]" \
  -e "$CODE/psytwill[dev]"

# Verify channel purity on the conda side (the FSL condarc trap again).
echo
echo "==> verifying conda channel purity"
strays="$(conda list --prefix "$PREFIX" --show-channel-urls 2>/dev/null \
          | grep -vE '^#' | awk 'NF{print $NF}' \
          | grep -viE 'conda-forge|^pypi$' || true)"
if [[ -n "$strays" ]]; then
  echo "FAIL: packages from a channel other than conda-forge:" >&2
  echo "$strays" | sort | uniq -c >&2
  exit 1
fi
echo "ok — conda-forge (+ pypi payload) only"

# Import check, from INSIDE the prefix (a resolved dep is not a verified
# one, and a check that doesn't assert the origin can silently report
# ~/.local's versions — both bites documented in duckbrain's script).
echo
"$PY" - "$PREFIX" <<'PYCHECK'
import sys
prefix = sys.argv[1]
assert sys.executable.startswith(prefix), sys.executable
import viz2psy, aud2psy, word2psy, psytwill              # the family
import numpy, torch, torchvision, transformers, gensim   # the shared core
import open_clip, faster_whisper, librosa, ultralytics
import fasttext, deepgaze_pytorch, cv2, sentence_transformers
print('python        ', sys.version.split()[0])
print('torch         ', torch.__version__, '| cuda', torch.version.cuda)
bad = []
editable = {'viz2psy', 'aud2psy', 'word2psy', 'psytwill'}
for m in (viz2psy, aud2psy, word2psy, psytwill, numpy, torch, torchvision,
          transformers, gensim, open_clip, faster_whisper, librosa,
          ultralytics, fasttext, deepgaze_pytorch, cv2, sentence_transformers):
    where = getattr(m, '__file__', '') or ''
    print(f'{m.__name__:<14}', getattr(m, '__version__', 'unknown'))
    if m.__name__ not in editable and not where.startswith(prefix):
        bad.append(f'{m.__name__} imported from {where}')
if bad:
    print('\nFAIL: modules resolved from outside the environment:', file=sys.stderr)
    print('\n'.join(bad), file=sys.stderr)
    sys.exit(1)
# The flagged §10 risk: gensim binary compatibility with numpy 2.x. Exercise
# it rather than trusting the import.
from gensim.models import KeyedVectors  # noqa: F401
kv_ok = numpy.__version__.split('.')[0]
print()
print(f'smoke: gensim {gensim.__version__} loads against numpy {numpy.__version__}')
PYCHECK

# Lock manifests: conda side (explicit URLs) + pip side (everything else).
# Together with pyproject.toml files these reproduce the env.
mkdir -p "$OWNER/conda"
CONDA_LOCK="$OWNER/conda/lock-linux-64.txt"
PIP_LOCK="$OWNER/conda/pip-lock-linux-64.txt"
echo
echo "==> writing $CONDA_LOCK"
{
  echo "# stimfeat conda layer, resolved by scripts/setup_env.sh against a fresh"
  echo "# prefix. Recreate: conda create --prefix <p> --file conda/lock-linux-64.txt"
  echo "# then the pip layer per conda/pip-lock-linux-64.txt."
  conda list --prefix "$PREFIX" --explicit
} > "$CONDA_LOCK"
echo "==> writing $PIP_LOCK"
{
  echo "# stimfeat pip layer (the payload), one resolve by scripts/setup_env.sh."
  echo "# The four family packages are EDITABLE installs from $CODE and appear"
  echo "# here at their versions on lock day. Recreate the layer with:"
  echo "#   pip install --extra-index-url $TORCH_INDEX \\"
  echo "#     '<deepgaze pin — see setup_env.sh>' -e <repo>[dev] x4"
  echo "# This freeze is the tested version standard for the family (contracts §10)."
  "$PY" -m pip list --format=freeze
} > "$PIP_LOCK"
echo "    $(grep -cv '^#' "$PIP_LOCK") pip packages pinned"

# activate.d hooks: user-site guard + shared caches for every
# `conda activate` user, not just this script.
mkdir -p "$PREFIX/etc/conda/activate.d" "$PREFIX/etc/conda/deactivate.d"
cat > "$PREFIX/etc/conda/activate.d/stimfeat-env.sh" <<HOOK
# Written by psytwill/scripts/setup_env.sh.
# 1) ~/.local site-packages shadow a conda env's own — disable user site.
# 2) Model caches are shared per-PIRG (multi-GB, identical per user).
# 3) HF_HOME is shared and group-readable, so huggingface_hub would derive a
#    *shared* default token file from it and a plain \`huggingface-cli login\`
#    would expose a personal token to the whole PIRG. Point HF_TOKEN_PATH at
#    the user's own home file; \$HOME is left unexpanded so it is per-user.
export _STIMFEAT_SAVED_PYTHONNOUSERSITE="\${PYTHONNOUSERSITE:-}"
export _STIMFEAT_SAVED_HF_HOME="\${HF_HOME:-}"
export _STIMFEAT_SAVED_HF_TOKEN_PATH="\${HF_TOKEN_PATH:-}"
export _STIMFEAT_SAVED_TORCH_HOME="\${TORCH_HOME:-}"
export _STIMFEAT_SAVED_WORD2PSY_CACHE="\${WORD2PSY_CACHE:-}"
export _STIMFEAT_SAVED_NLTK_DATA="\${NLTK_DATA:-}"
export _STIMFEAT_SAVED_YOLO_CONFIG_DIR="\${YOLO_CONFIG_DIR:-}"
export PYTHONNOUSERSITE=1
export HF_HOME=$CACHE/huggingface
export HF_TOKEN_PATH="\${HF_TOKEN_PATH:-\$HOME/.cache/huggingface/token}"
export TORCH_HOME=$CACHE/torch
export WORD2PSY_CACHE=$CACHE/word2psy
export NLTK_DATA=$CACHE/nltk_data
export YOLO_CONFIG_DIR=$CACHE/ultralytics
HOOK
cat > "$PREFIX/etc/conda/deactivate.d/stimfeat-env.sh" <<'HOOK'
for v in PYTHONNOUSERSITE HF_HOME HF_TOKEN_PATH TORCH_HOME WORD2PSY_CACHE NLTK_DATA YOLO_CONFIG_DIR; do
  saved="_STIMFEAT_SAVED_$v"
  if [ -n "${!saved:-}" ]; then export "$v=${!saved}"; else unset "$v"; fi
  unset "$saved"
done
HOOK

# Record the prefix in each family repo (gitignored, like duckbrain's).
for r in "${REPOS[@]}"; do
  echo "$PREFIX" > "$CODE/$r/.conda-prefix"
  grep -qx '.conda-prefix' "$CODE/$r/.gitignore" 2>/dev/null \
    || echo '.conda-prefix' >> "$CODE/$r/.gitignore"
done
echo
echo "recorded prefix in ${REPOS[*]/#/.conda-prefix @ }"

# Group permissions: umask 0002 + inherited setgid cover the common case;
# sweep for anything pip installed with restrictive modes anyway.
echo "==> group-permission sweep (g+rwX over prefix and cache)"
chmod -R g+rwX "$PREFIX" "$CACHE" "$ENVS/bin" 2>/dev/null || true

cat <<EOF

Activate with:
  module load $CONDA_MODULE
  conda activate $PREFIX

The activate hook exports PYTHONNOUSERSITE=1 and the shared caches
(HF_HOME, TORCH_HOME, WORD2PSY_CACHE, NLTK_DATA, YOLO_CONFIG_DIR), and
points HF_TOKEN_PATH at \$HOME/.cache/huggingface/token so HF credentials
stay per-user rather than landing in the shared cache.
EOF
