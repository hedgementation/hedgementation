#!/usr/bin/env bash
set -euo pipefail

EXTRACT_DIR="${1:-.}"
mkdir -p "$EXTRACT_DIR"

wget --progress=bar:force:noscroll -O AEZ_LR.zip \
    "https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/LR.zip"
wget --progress=bar:force:noscroll -O AEZ_symbology.zip \
    "https://gaez-v4-data.fao.org/data/documentation/GAEZ4_symbology_files.zip"

7za x AEZ_LR.zip -o"$EXTRACT_DIR"
7za x AEZ_symbology.zip -o"$EXTRACT_DIR"