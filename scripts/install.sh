#!/bin/sh
# Termite install script
# Based on the Antfly install script approach
set -eu

status() { echo ">>> $*" >&2; }
error() { echo "ERROR: $*" >&2; exit 1; }
warning() { echo "WARNING: $*" >&2; }

TEMP_DIR=$(mktemp -d)
cleanup() { rm -rf "$TEMP_DIR"; }
trap cleanup EXIT

available() { command -v "$1" >/dev/null; }

require() {
    local MISSING=''
    for TOOL in "$@"; do
        if ! available "$TOOL"; then
            MISSING="$MISSING $TOOL"
        fi
    done

    if [ -n "$MISSING" ]; then
        error "Missing required tools:$MISSING. Please install them and try again."
    fi
}

# Detect OS and architecture
detect_platform() {
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)

    case "$OS" in
        linux) OS="Linux" ;;
        darwin) OS="Darwin" ;;
        *) error "Unsupported operating system: $OS" ;;
    esac

    case "$ARCH" in
        x86_64|amd64) ARCH="x86_64" ;;
        aarch64|arm64) ARCH="arm64" ;;
        *) error "Unsupported architecture: $ARCH" ;;
    esac

    # Darwin uses universal binaries
    if [ "$OS" = "Darwin" ]; then
        ARCH="all"
    fi

    echo "$OS $ARCH"
}

# Download and install termite
install_termite() {
    require curl tar

    status "Detecting platform..."
    read -r OS ARCH <<EOF
$(detect_platform)
EOF
    status "Detected platform: $OS $ARCH"

    VERSION="${1:-latest}"

    # Handle 'latest' version
    if [ "$VERSION" = "latest" ]; then
        status "Fetching latest version..."
        # Try to get version from metadata
        LATEST_URL="https://releases.antfly.io/termite/latest/metadata.json"
        if VERSION_INFO=$(curl -fsSL "$LATEST_URL" 2>/dev/null); then
            VERSION=$(echo "$VERSION_INFO" | grep -o '"tag":"[^"]*"' | head -1 | cut -d'"' -f4)
        fi
        if [ -z "$VERSION" ] || [ "$VERSION" = "latest" ]; then
            error "Could not determine latest version. Please specify a version explicitly."
        fi
    fi

    # Normalize version: TAG has termite-v prefix, VERSION_NUM does not
    # GoReleaser uses .Tag (with prefix) for paths and .Version (without prefix) for filenames
    case "$VERSION" in
        termite-v*) TAG="$VERSION"; VERSION_NUM="${VERSION#termite-v}" ;;
        v*) TAG="termite-$VERSION"; VERSION_NUM="${VERSION#v}" ;;
        *)  TAG="termite-v$VERSION"; VERSION_NUM="$VERSION" ;;
    esac

    status "Installing Termite version $TAG..."

    # Construct download URL
    # Format: https://releases.antfly.io/termite/termite-v0.1.0/termite_0.1.0_Darwin_all.tar.gz
    ARCHIVE_NAME="termite_${VERSION_NUM}_${OS}_${ARCH}.tar.gz"
    DOWNLOAD_URL="https://releases.antfly.io/termite/${TAG}/${ARCHIVE_NAME}"

    status "Downloading from $DOWNLOAD_URL..."
    if ! curl -fsSL "$DOWNLOAD_URL" -o "$TEMP_DIR/$ARCHIVE_NAME"; then
        error "Failed to download Termite. Please check your internet connection and the version number."
    fi

    status "Extracting archive..."
    tar -xzf "$TEMP_DIR/$ARCHIVE_NAME" -C "$TEMP_DIR"

    # Determine install location
    if [ "$(id -u)" -eq 0 ]; then
        # Running as root
        INSTALL_DIR="/usr/local/bin"
    else
        # Running as regular user
        INSTALL_DIR="$HOME/.local/bin"
        mkdir -p "$INSTALL_DIR"
    fi

    status "Installing binary to $INSTALL_DIR..."

    # Install termite
    if [ -f "$TEMP_DIR/termite" ]; then
        if [ -w "$INSTALL_DIR" ] || [ "$(id -u)" -eq 0 ]; then
            mv "$TEMP_DIR/termite" "$INSTALL_DIR/termite"
            chmod +x "$INSTALL_DIR/termite"
            status "Installed termite to $INSTALL_DIR/termite"
        else
            status "Need sudo permission to install to $INSTALL_DIR"
            sudo mv "$TEMP_DIR/termite" "$INSTALL_DIR/termite"
            sudo chmod +x "$INSTALL_DIR/termite"
            status "Installed termite to $INSTALL_DIR/termite"
        fi
    else
        error "termite binary not found in archive"
    fi

    status "Termite installation complete!"
    status ""
    status "Run 'termite --help' to get started"

    # Check if install dir is in PATH
    case ":$PATH:" in
        *":$INSTALL_DIR:"*) ;;
        *)
            warning "$INSTALL_DIR is not in your PATH"
            warning "Add the following to your shell profile (~/.bashrc, ~/.zshrc, etc.):"
            warning "  export PATH=\"\$PATH:$INSTALL_DIR\""
            ;;
    esac
}

# Main execution
main() {
    if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
        cat <<EOF
Termite Installer

Usage:
  curl -fsSL https://releases.antfly.io/termite/latest/install.sh | sh
  curl -fsSL https://releases.antfly.io/termite/latest/install.sh | sh -s -- termite-v0.1.0

Options:
  -h, --help    Show this help message
  [version]     Install a specific version (e.g., termite-v0.1.0, v0.1.0, or 0.1.0)

Environment:
  This script will automatically detect your OS and architecture,
  download the appropriate binary, and install it.

  By default, it installs to:
    - /usr/local/bin (if running as root)
    - ~/.local/bin (if running as regular user)

Termite is a standalone ML inference service for embeddings, chunking, and reranking.

For more information, visit: https://docs.antfly.io/docs/guides/termite
EOF
        exit 0
    fi

    install_termite "$@"
}

main "$@"
