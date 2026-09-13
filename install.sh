#!/usr/bin/env bash
# cuda-ir.go dependency installer: LLVM 22, llgo + llgen, the gocuda command, LLGO_ROOT.
#
#   ./install.sh            # interactive (asks before sudo / editing your profile)
#   ./install.sh -y         # no questions
#
# Environment overrides:
#   LLGO_ROOT   where to clone llgo            (default: $HOME/llgo)
#   LLGO_REPO   llgo git URL or local path     (default: https://github.com/xgo-dev/llgo.git)
#   LLGO_REF    llgo commit/tag to check out   (default: tested commit below)
#   LLVM_VER    LLVM major version             (default: 22)
#   GOCUDA_REF  cuda-ir.go version for go install  (default: this checkout if run from the repo, else @latest)
set -euo pipefail

LLVM_VER=${LLVM_VER:-22}
LLGO_ROOT=${LLGO_ROOT:-$HOME/llgo}
LLGO_REPO=${LLGO_REPO:-https://github.com/xgo-dev/llgo.git}
LLGO_REF=${LLGO_REF:-4606197d8}          # tested with cuda-ir.go; "main" for latest
GOCUDA_MODULE=github.com/mehdi-shokohi/cuda-ir.go
YES=0
[[ "${1:-}" == "-y" ]] && YES=1

say()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32mok\033[0m  %s\n' "$*"; }
warn() { printf '    \033[33m!!\033[0m  %s\n' "$*"; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
ask()  { [[ $YES == 1 ]] && return 0; read -r -p "    $1 [Y/n] " a; [[ -z $a || $a =~ ^[Yy] ]]; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OS=$(uname -s)
GOBIN_DIR=$(go env GOBIN 2>/dev/null || true); GOBIN_DIR=${GOBIN_DIR:-$(go env GOPATH 2>/dev/null)/bin}
export PATH="$PATH:$GOBIN_DIR"

# ---------------------------------------------------------------- 1. Go
say "Go"
command -v go >/dev/null || die "Go is not installed: https://go.dev/dl (1.24 or newer)"
gover=$(go env GOVERSION | sed 's/^go//')
ok "go $gover ($(command -v go))"
case $gover in 1.[0-9].*|1.1[0-9].*|1.2[0-3].*) die "Go 1.24+ required, found $gover";; esac

# ---------------------------------------------------------------- 2. LLVM
say "LLVM $LLVM_VER (llvm-link / opt / llc with the NVPTX backend; llgo also needs the dev libraries)"
have_llvm() { command -v "llc-$LLVM_VER" >/dev/null || command -v llc >/dev/null; }
if [[ $OS == Linux ]] && command -v apt-get >/dev/null; then
    # what llgen (cgo -> libLLVM, libffi) and the LLVM tools need; the full llgo
    # compiler additionally wants libgc-dev libssl-dev libcjson-dev libsqlite3-dev
    # libuv1-dev libunwind-$LLVM_VER-dev libc++-$LLVM_VER-dev (see llgo's README)
    pkgs=(llvm-$LLVM_VER-dev clang-$LLVM_VER lld-$LLVM_VER pkg-config libffi-dev zlib1g-dev)
    missing=()
    for p in "${pkgs[@]}"; do dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p"); done
    if [[ ${#missing[@]} -eq 0 ]]; then
        ok "all apt packages present"
    else
        warn "missing: ${missing[*]}"
        if ask "Install with apt.llvm.org's llvm.sh + apt-get (needs sudo)?"; then
            if ! have_llvm; then
                tmp=$(mktemp -d); wget -qO "$tmp/llvm.sh" https://apt.llvm.org/llvm.sh; chmod +x "$tmp/llvm.sh"
                sudo "$tmp/llvm.sh" "$LLVM_VER"      # adds the apt.llvm.org repo for this distro and installs llvm-22
            fi
            sudo apt-get install -y "${missing[@]}"
        else
            die "LLVM $LLVM_VER is required"
        fi
    fi
elif [[ $OS == Darwin ]]; then
    if brew list llvm@$LLVM_VER >/dev/null 2>&1; then ok "brew llvm@$LLVM_VER present"; else
        ask "brew install llvm@$LLVM_VER lld@$LLVM_VER bdw-gc openssl cjson libffi libuv pkg-config?" || die "LLVM required"
        brew install llvm@$LLVM_VER lld@$LLVM_VER bdw-gc openssl cjson libffi libuv pkg-config
        brew link --force --overwrite llvm@$LLVM_VER lld@$LLVM_VER libffi
    fi
    export LLVM_SUFFIX=""
else
    have_llvm || die "install LLVM $LLVM_VER for your platform (llvm-link, opt, llc), then re-run"
    ok "found $(command -v llc-$LLVM_VER 2>/dev/null || command -v llc)"
fi
llc_bin=$(command -v llc-$LLVM_VER 2>/dev/null || command -v llc)
"$llc_bin" --version | grep -q nvptx || die "$llc_bin has no NVPTX backend"
ok "$llc_bin has the nvptx64 target"

# ---------------------------------------------------------------- 3. llgo + llgen
say "llgo checkout at $LLGO_ROOT (llgen compiles Go to LLVM IR and needs this tree at run time)"
if [[ -d $LLGO_ROOT/.git ]]; then
    ok "exists; fetching"
    git -C "$LLGO_ROOT" fetch -q origin
else
    git clone -q "$LLGO_REPO" "$LLGO_ROOT"
fi
git -C "$LLGO_ROOT" checkout -q "$LLGO_REF" 2>/dev/null || git -C "$LLGO_ROOT" checkout -q "origin/$LLGO_REF"
ok "llgo at $(git -C "$LLGO_ROOT" describe --tags --always)"
say "building llgen"
(cd "$LLGO_ROOT" && go install ./chore/llgen)
ok "$(command -v llgen)"
export LLGO_ROOT

# ---------------------------------------------------------------- 4. gocuda
say "gocuda"
if [[ -f $SCRIPT_DIR/go.mod ]] && grep -q "^module $GOCUDA_MODULE" "$SCRIPT_DIR/go.mod"; then
    (cd "$SCRIPT_DIR" && go install ./cmd/gocuda)
else
    go install "$GOCUDA_MODULE/cmd/gocuda@${GOCUDA_REF:-latest}"
fi
ok "$(command -v gocuda)"

# ---------------------------------------------------------------- 5. CUDA (checked, not installed)
say "CUDA"
found=""
for f in /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib64/libcuda.so.1 /usr/lib/libcuda.so.1 /usr/lib/aarch64-linux-gnu/libcuda.so.1; do
    [[ -e $f ]] && found=$f && break
done
if [[ -n $found ]]; then
    ok "NVIDIA driver ($found)"
else
    warn "NVIDIA driver not found - needed to run kernels: https://www.nvidia.com/drivers"
fi
found=""
for f in "${LIBDEVICE:-}" "${CUDA_HOME:-/nonexistent}"/nvvm/libdevice/libdevice.10.bc /usr/local/cuda/nvvm/libdevice/libdevice.10.bc /usr/local/cuda-*/nvvm/libdevice/libdevice.10.bc /opt/cuda/nvvm/libdevice/libdevice.10.bc; do
    [[ -n $f && -e $f ]] && found=$f && break
done
if [[ -n $found ]]; then
    ok "CUDA toolkit ($found)"
else
    warn "CUDA toolkit not found - needed for cuda.Sin/Exp/... (libdevice) and the ptxas check:"
    warn "https://developer.nvidia.com/cuda-downloads  (then: export CUDA_HOME=/usr/local/cuda)"
fi

# ---------------------------------------------------------------- 6. shell profile
say "environment"
profile=$HOME/.bashrc; [[ ${SHELL:-} == */zsh ]] && profile=$HOME/.zshrc
lines=("export LLGO_ROOT=\"$LLGO_ROOT\"" "export PATH=\"\$PATH:$GOBIN_DIR\"")
[[ $OS == Darwin ]] && lines+=('export LLVM_SUFFIX=""')
need=()
for l in "${lines[@]}"; do grep -qxF "$l" "$profile" 2>/dev/null || need+=("$l"); done
if [[ ${#need[@]} -gt 0 ]]; then
    if ask "Append to $profile: ${need[*]} ?"; then
        { echo; echo "# cuda-ir.go"; printf '%s\n' "${need[@]}"; } >> "$profile"
        ok "written; run:  source $profile"
    else
        warn "add these yourself: ${need[*]}"
    fi
else
    ok "$profile already set"
fi

# ---------------------------------------------------------------- 7. verify
say "gocuda doctor"
gocuda doctor
