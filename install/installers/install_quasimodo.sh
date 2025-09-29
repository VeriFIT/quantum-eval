#!/bin/bash
set -e
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../configs/quasimodo.conf"
MAKEFILE="$SCRIPT_DIR/../misc/Makefile"
log "Installing Quasimodo..."

# Check dependencies
for dep in "${DEPS[@]}"; do
    if ! dpkg -s "$dep" >/dev/null 2>&1; then
        log "Installing $dep"
        sudo apt-get install -y "$dep"
    else
        log "$dep is already installed"
    fi
done

# Check Python dependencies
for pipdep in "${PIPDEPS[@]}"; do
    if ! pip show "$pipdep" >/dev/null 2>&1; then
        log "Installing Python dependency: $pipdep"
        pip install --quiet "$pipdep"
    else
        log "Python dependency $pipdep is already installed"
    fi
done

# Clone repo
clone_repo "$REPO" "$DIR"

log Cloning Quasimodo backend...
git clone https://github.com/trishullab/Quasimodo.git "$DIR"/Quasimodo
cd "$DIR"/Quasimodo/
git rm -f MQT_DD/dd_package || true
log Copying correct Makefile...
rm Makefile
cp "$MAKEFILE" "$DIR"/Quasimodo/
git submodule update --init

log Downloading supported version of boost...
# Download Boost 1.81 (tar.bz2)
wget -nc https://archives.boost.io/release/1.81.0/source/boost_1_81_0.tar.gz

# Extract
tar -xzf boost_1_81_0.tar.gz

# Set environment variable
export BOOST_PATH="$(pwd)/boost_1_81_0"

cd cflobdd/cudd-complex-big/
autoupdate
autoreconf

sed -i 's/: ${CFLAGS="-Wall -Wextra -g -O3"}/: ${CFLAGS="-Wall -Wextra -g -O3 -fPIC"}/g' configure
sed -i 's/: ${CXXFLAGS="-Wall -Wextra -std=c++0x -g -O3"}/: ${CXXFLAGS="-Wall -Wextra -std=c++0x -g -O3 -fPIC"}/g' configure

./configure

make
cd ../..

cd python_pkg/
invoke build-quasimodo

log Building QuasimodoSim...
cd "$DIR"
make
