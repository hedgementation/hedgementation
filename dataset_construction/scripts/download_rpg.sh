#!/usr/bin/env bash
set -euo pipefail

URL="https://data.geopf.fr/telechargement/download/RPG/RPG_2-0__SHP_LAMB93_R24_2022-01-01/RPG_2-0__SHP_LAMB93_R24_2022-01-01.7z.001"
ARCHIVE="RPG_archive.7z"

EXTRACT_DIR="${1:-.}"
mkdir -p "$EXTRACT_DIR"

wget --progress=bar:force:noscroll -O "$ARCHIVE" "$URL"

7za x "$ARCHIVE" -o"$EXTRACT_DIR"