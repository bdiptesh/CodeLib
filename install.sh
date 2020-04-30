#!/bin/bash
# =============================================================================
# Ratings model installation file
#
# Version: 0.1.0
#
# Author: Diptesh.Basak
#
# Date: May 15, 2019
#
# =============================================================================

# =============================================================================
# DO NOT CHANGE ANYTHING FROM HERE
# =============================================================================

error=0

# =============================================================================
# User defined functions
# =============================================================================

mod()
{
  exec=$1
  file=$2
  printf "%-72s %s" "$2"
  if $exec $file; then
    state="[ OK ]"
  else
    state="[fail]"
    error=$((error + 1))
  fi
  printf "$state\n"
}

# =============================================================================
# Main
# =============================================================================

path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)\
/$(basename "${BASH_SOURCE[0]}")"

proj_dir=$(sed -E 's/(.+\/)(.+)/\1/' <<< $path)

for i in $(find "$proj_dir" -maxdepth 10 -name "*.sh")
do
  file_name=${i#$proj_dir}
  mod "chmod +x" "$file_name"
  if [[ "$file_name" == "programs.sh" ]]; then
    bash bin/programs.sh
  fi
done

mod "chmod +x" "install.sh"

printf "%-72s %s" "Installation"
if [[ $error -gt 0 ]]; then
  state="[fail]"
else
  state="[Done]"
fi
printf "$state\n"

exit $error
