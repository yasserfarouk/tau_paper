#!/usr/bin/env bash
year=${1}
shift

base="${SCRIPT_DIR}/${basefolder}"
donefolder="serverclean"
if [ ${year} -eq  0 ]; then
	donefolder="serverclean/final.csv"
else
	donefolder="serverclean/y${year}/y${year}.csv"
fi
python src/experiment.py --aoyear="${year}" --done=results --done="${donefolder}" --ao=none --year="${year}" --f0=-1 --f1=-1 --title="Per Year ${year}" --path="results/y${year}" --fast-start --trials=1 --n-retries=3 --timelimit-final-retry --no-ignore-common-failures --genius10 --add-cab-war "$@"
