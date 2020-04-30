#!/bin/bash
# =============================================================================
# To refresh data
#
# Version: 0.1.0
#
# Author: Diptesh
#
# Date: Apr 08, 2019
# =============================================================================

# =============================================================================
# DO NOT CHANGE ANYTHING FROM HERE
# =============================================================================

__version__=0.1.0

# =============================================================================
# User defined functions
# =============================================================================

prd_cube () {
  printf -- '-%.0s' {1..72}
  printf "\nRunning Product cube..."
  printf -- '-%.0s' {1..72}
  hive --hivevar ty_start_d=$ty_start_d \
  --hivevar ty_end_d=$ty_end_d -f $path/hive/prd_cube.hql \
  > $path/log/prd_cube.log
}

raw_mod_dat () {
  printf -- '-%.0s' {1..72}
  printf "\nRunning Raw model data..."
  printf -- '-%.0s' {1..72}
  hive -f $path/hive/raw_mod_dat.hql > $path/log/raw_mod_dat.log
}

mod_dat () {
  printf -- '-%.0s' {1..72}
  printf "\nRunning Model data..."
  printf -- '-%.0s' {1..72}
  hive -f $path/hive/mod_dat.hql > $path/log/mod_dat.log
}

str_attr () {
  printf -- '-%.0s' {1..72}
  printf "\nRunning Store attributes..."
  printf -- '-%.0s' {1..72}
  hive --hivevar ty_start_d=$ty_start_d \
  --hivevar ty_end_d=$ty_end_d \
  --hivevar ly_start_d=$ly_start_d \
  --hivevar ly_end_d=$ly_end_d \
  -f $path/hive/str_attr.hql > $path/log/str_attr.log
}

# =============================================================================
# Main
# =============================================================================

# Set path to root directory

path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)\
/$(basename "${BASH_SOURCE[0]}")"
path=$(sed -E 's/(.+)(\/bin\/.+)/\1/' <<< $path)

printf -- '-%.0s' {1..72}
printf "\nRUNNING $(basename $0) v$__version__"
printf "\nPATH: $path\n"
printf -- '-%.0s' {1..72}

# Set TY/LY start and end dates

printf "\nMODULE: Sales/Backroom table dates\n"

end_d1=`hive --hiveconf hive.session.id=LATEST_SALES_DATE \
        --hiveconf hive.execution.engine=tez -e\
        "select max(acct_wk_end_d)
        from ino_spd_fnd.sls_div_dim;"`

end_d2=`hive --hiveconf hive.session.id=LATEST_BKRM_DATE \
        --hiveconf hive.execution.engine=tez -e\
        "select max(acct_wk_end_d)
        from ino_spd_fnd.bkrm_loc_item_dim;"`

date_diff="$(echo $(($(($(date -d "${end_d1}" "+%s") - \
           $(date -d "${end_d2}" "+%s"))) / 86400)))"

if [[ $date_diff -gt 0 ]]; then
  end_d="$end_d2"
else
  end_d="$end_d1"
fi

ty_end_d=$(date -d "$end_d" +%F)
ty_start_d=$(date -d "$end_d 51 week ago" "+%Y-%m-%d")
ly_end_d=$(date -d "$end_d 52 week ago" "+%Y-%m-%d")
ly_start_d=$(date -d "$end_d 103 week ago" "+%Y-%m-%d")

printf "MODULE: TY Start date: $ty_start_d --- TY End date: $ty_end_d\n"
printf "MODULE: LY Start date: $ly_start_d --- LY End date: $ly_end_d\n"

# Run hive scripts

prd_cube
raw_mod_dat
mod_dat
str_attr

# EOF

printf -- '-%.0s' {1..72}
printf "\nCompleted running $(basename $0) v$__version__\n"
printf -- '-%.0s' {1..72}
