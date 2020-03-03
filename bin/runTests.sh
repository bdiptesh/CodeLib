#!/bin/bash
# =============================================================================
# Python unit tests and code ratings
#
# Objective: Test all python test modules and rate all python scripts
#
# Version: 0.1.0
#
# Author: Diptesh
#
# Date: Mar 03, 2020
#
# =============================================================================

# Set test directory
path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)\
/$(basename "${BASH_SOURCE[0]}")"

test_dir=$(sed -E 's/(.+)(\/bin\/.+)/\1\/test/' <<< $path)
proj_dir=$(sed -E 's/(.+)(\/bin\/.+)/\1/' <<< $path)

if [[ -z $1 ]]
then
	module="-a"
else
	module=$1
fi

printf "=%.0s" {1..70}

# Run unit tests
if [[ $module == "-a" || $module == "-u" ]]
then
    printf "\nRunning unit tests...\n\n"
    python -m unittest discover -v -s $test_dir -p "test_*.py"
    printf "=%.0s" {1..70}
    printf "\n"
fi

# Rate coding styles for all python scripts
if [[ $module == "-a" || $module == "-r" ]]
then
    printf "\nRating code style...\n\n"
    score=0
    cnt=0
    for i in $(find "$proj_dir" -name "*.py")
    do
        file=${i#$(dirname "$(dirname "$i")")/}
        printf "%-67s %s" "$file"
        file_dir=$(sed -E 's/(.+\/)(.+\.py)/\1/' <<< $i)
        cd "$file_dir"
        pylint "$i" > "$proj_dir/log/pylint/pylint.out"
        PYLINT_SCORE=`grep "Your code has been rated" $proj_dir/log/pylint/pylint.out | cut -d" " -f7 | cut -d"." -f1`
        file_name=$(sed -E 's/(\/)/-/' <<< $file)
        file_name=$(sed -E 's/(\.)/-/' <<< $file_name)
        cp "$proj_dir/log/pylint/pylint.out" "$proj_dir/log/pylint/$file_name.out" 
        score=$((score + PYLINT_SCORE))
        cnt=$((cnt + 1))
        printf "$PYLINT_SCORE\n"
        cd "$proj_dir"
    done
    tot_score=$(echo "scale=1; $score/$cnt" | bc)
    printf "\nTotal score: $tot_score\n"
    # Add pylint badge to README.md
    sed -i "1s/.*/\!\[pylint Score\]\(https\:\/\/mperlet\.github\.io\/pybadge\/badges\/$tot_score.svg\)/" "/home/ph33r/Project/CodeLib/README.md"
    printf "=%.0s" {1..70}
    printf "\n"
fi

exit 0
