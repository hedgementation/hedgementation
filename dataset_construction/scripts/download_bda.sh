#!/usr/bin/env bash
set -euo pipefail

URL="https://data.geopf.fr/telechargement/download/BD-HAIE/HAIE_2-0__GPKG_LAMB93_FXX_2024-03-15/HAIE_2-0__GPKG_LAMB93_FXX_2024-03-15.7z"
ARCHIVE="BDA_archive.7z"

EXTRACT_DIR="${1:-.}"
mkdir -p "$EXTRACT_DIR"

wget --progress=bar:force:noscroll -O "$ARCHIVE" "$URL"

7za x "$ARCHIVE" -o"$EXTRACT_DIR"