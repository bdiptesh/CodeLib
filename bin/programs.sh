#!/bin/bash
# =============================================================================
# Python dependencies
#
# Objective: Install python dependencies from requirements.txt
# 
# Version: 0.3.0
#
# Author: Diptesh.Basak
#
# Date: Jun 02, 2020
#
# =============================================================================

# Set project directory
path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)\
/$(basename "${BASH_SOURCE[0]}")"

proj_dir=$(sed -E 's/(.+)(\/bin\/.+)/\1/' <<< $path)

pip install -r $proj_dir/requirements.txt
