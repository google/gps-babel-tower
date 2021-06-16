#!/bin/bash

set -euo pipefail

python3 -m build
python3 -m twine upload dist/*
rm -rf build dist