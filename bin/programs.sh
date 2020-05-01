#!/bin/bash
# =============================================================================
# Create shared library files from C programs
#
# Objective: Test all python test modules
#
# Version: 0.1.0
#
# Author: Diptesh
#
# Date: Sep 02, 2019
#
# =============================================================================

# Set bin directory

path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)\
/$(basename "${BASH_SOURCE[0]}")"

proj_dir=$(sed -E 's/(.+\/)(.+)/\1/' <<< $path)

prog=$(sed -E 's/(.+\/)(.+)/\2/' <<< $path)


printf "=%.0s" {1..79}
printf "\nRunning $prog ...\n"
printf "\nCreating shared library files ...\n\n"

for i in $(find "$proj_dir" -name "*.c")
do
  file_name=$(sed -E 's/(.+\/)(.+)(\.c)/\2/' <<< $i)
  printf "%-72s %s" "$file_name.c" "[ OK ]"
  # printf "$proj_dir --- $path --- $file_name"
  printf "\n"
  cc -fPIC -shared -o $proj_dir$file_name.so $proj_dir$file_name.c
done

printf "=%.0s" {1..79}
printf "\n"