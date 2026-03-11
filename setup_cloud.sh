#!/usr/bin/env bash

set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
pip install -r pathfinder/requirements_cloud.txt
pip install -r requirements_cloud.txt
