#!/usr/bin/env bash
# bluellm installer
# Usage: curl -fsSL https://raw.githubusercontent.com/aokumablue/bluellm/main/install.sh | bash
#        bash install.sh
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

REPO_URL="https://github.com/aokumablue/bluellm.git"
INSTALL_DIR="${BLUELLM_DIR:-$HOME/.bluellm}"

_die() { tput cnorm 2>/dev/null || true; echo -e "${YELLOW}error: $*${NC}" >&2; exit 1; }

_spinner() {
    local pid=$1
    local msg="${2:-working...}"
    local frames=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
    local cyan="\033[1;36m" reset="\033[0m"

    if ! [ -t 2 ]; then
        wait "$pid"
        return $?
    fi

    tput civis 2>/dev/null || true
    while kill -0 "$pid" 2>/dev/null; do
        for frame in "${frames[@]}"; do
            printf "\r%b%s%b %s" "$cyan" "$frame" "$reset" "$msg" >&2
            sleep 0.08
        done
    done
    printf "\r\033[K" >&2
    tput cnorm 2>/dev/null || true
}

trap 'printf "\r\033[K" >&2; tput cnorm 2>/dev/null || true' EXIT INT TERM

_LOCAL_DIR=""
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
    _LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

_setup_repo() {
    if [ -n "$_LOCAL_DIR" ] && [ -f "$_LOCAL_DIR/pyproject.toml" ]; then
        cd "$_LOCAL_DIR"
        return
    fi

    if [ -d "$INSTALL_DIR/.git" ]; then
        origin_url="$(git -C "$INSTALL_DIR" remote get-url origin 2>/dev/null || true)"
        case "$origin_url" in
            *aokumablue/bluellm|*aokumablue/bluellm.git)
                git -C "$INSTALL_DIR" pull --ff-only origin main >/dev/null 2>&1 &
                _spinner $! "Updating..."
                wait $! || _die "git pull failed"
                ;;
            *)
                _die "unexpected origin at $INSTALL_DIR: $origin_url"
                ;;
        esac
    elif [ -d "$INSTALL_DIR" ]; then
        _die "$INSTALL_DIR exists but is not a bluellm install"
    else
        command -v git >/dev/null 2>&1 || _die "git is required"
        git clone --depth 1 "$REPO_URL" "$INSTALL_DIR" >/dev/null 2>&1 &
        _spinner $! "Cloning..."
        wait $! || _die "git clone failed"
    fi
    cd "$INSTALL_DIR"
}

_setup_repo

# ── venv & install ───────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -e . -q &
_spinner $! "Installing..."
wait $! || _die "pip install failed"

# ── generate keys ────────────────────────────────────────────────────────────
MASTER_KEY=$(openssl rand -base64 24)
SALT_KEY=$(openssl rand -base64 24)

# ── read secret (masked input) ───────────────────────────────────────────────
_read_secret() {
    local prompt="$1"
    local _secret="" _char
    printf "%s" "$prompt"
    exec 3< /dev/tty
    while IFS= read -r -s -n1 -u3 _char; do
        if [[ $_char == $'\0' || $_char == $'\n' || $_char == $'\r' ]]; then
            break
        elif [[ $_char == $'\177' || $_char == $'\b' ]]; then
            if [ ${#_secret} -gt 0 ]; then
                _secret="${_secret%?}"
                printf '\b \b'
            fi
        else
            _secret+="$_char"
            printf '*'
        fi
    done
    exec 3<&-
    echo
    SECRET_REPLY="$_secret"
}

# ── user input ───────────────────────────────────────────────────────────────
echo
printf "endpoint: "
read -r ENDPOINT < /dev/tty

VERSION="${VERSION:-2025-04-01-preview}"

_read_secret "key: "
AZURE_API_KEY="$SECRET_REPLY"

if [ -z "$ENDPOINT" ] || [ -z "$AZURE_API_KEY" ]; then
    _die "endpoint and API key are required"
fi

# ── write .env ───────────────────────────────────────────────────────────────
cat > .env << EOF
BLUELLM_MASTER_KEY=$MASTER_KEY
BLUELLM_SALT_KEY=$SALT_KEY
EOF
chmod 600 .env

# ── encrypt key & write config.yml ──────────────────────────────────────────
export BLUELLM_SALT_KEY="$SALT_KEY"
ENCRYPTED=$(BLUELLM_VALUE_TO_ENCRYPT="$AZURE_API_KEY" python3 - << 'PYEOF'
import os, sys
sys.path.insert(0, "src")
from bluellm import crypto
value = os.environ["BLUELLM_VALUE_TO_ENCRYPT"]
salt_key = os.environ["BLUELLM_SALT_KEY"]
token = crypto.encrypt_value(value, salt_key)
print(f"encrypted:{token}")
PYEOF
)

cat > config.yml << EOF
models:
  - name: "*"
    params:
      model: azure/gpt-5.4
      endpoint: $ENDPOINT
      key: $ENCRYPTED
      version: $VERSION

generals:
  key: os.environ/BLUELLM_MASTER_KEY
  salt: os.environ/BLUELLM_SALT_KEY
  host: 127.0.0.1
  port: 8888
EOF

echo -e "\n${GREEN}✓ bluellm installed${NC}"
