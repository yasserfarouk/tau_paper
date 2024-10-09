#!/usr/bin/env bash
# compiles dst_file from various runs together
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
usage() {                                 # Function: Print a help message.
    echo "Usage: $("basename $0") [-s source] [-d destination] [-p per-year years] [-P] [-u] [-n]"
    echo "   -p: Years to use for per year. Pass as a spaces separated list. Default is 2010 to 2016 except 2014"
    echo "   -P: do not compile per year data"
    echo "   -d: destination folder to save everything"
    echo "   -s: source for data"
    echo "   -u: Just update instead of recopmiling"
    echo "   -r: remove explicit failures"
    echo "   -i: remove implicit failures"
    echo "   -n: # outcomes limit"
    echo "   -c: complete missing AO"
    echo "   -C: complete missing TAU"
    echo "   -t: Minimum number of trials to keep for each run condition (even using resampling)"
    echo "   -T: Maximum number of trials to keep for each run condition (any more than that will be dropped)"
    echo "   -v: verbose"
    echo "   -N: keep finalists only"
    echo "   -G: keep Gneius10 only"


}
exit_abnormal() {
    usage
    exit 1
}
remove_nonfinalists=""
remove_nongenius10=""
srcbase="server"
dstbase="serverclean"
peryear="2010 2011 2012 2013 2015 2016 2017 2018"
trials=1
max_trials=1
update="yes"
remove_explicit="--no-remove-explicit-failures"
remove_implicit="--no-remove-implicit-failures"
remove_negative="--no-remove-negative-advantage"
outcomelimit=0
complete_tau="--no-complete-tau"
complete_all="--no-complete-missing-all"
complete_some="--no-complete-missing-some"
excluded_domains=""
verbose="--no-verbose"
unchecked_strategies=""

while getopts "U:GNAvSACriuPn:p:d:s:t:T:" options; do
    case "${options}" in
        n) outcomelimit="${OPTARG}" ;;
        d) dstbase="${OPTARG}" ;;
        s) srcbase="${OPTARG}" ;;
        t) trials="${OPTARG}" ;;
        T) max_trials="${OPTARG}" ;;
        p) peryear="${OPTARG}" ;;
        U) unchecked_strategies="--excluded=${OPTARG}" ;;
        P) peryear="" ;;
        N) remove_nonfinalists="yes" ;;
        G) remove_nongenius10="yes" ;;
        A) remove_negative="--remove-negative-advantage" ;;
        v) verbose="--verbose" ;;
        C) complete_tau="--complete-tau" ;;
        S) complete_some="--complete-missing-some" ;;
        A) complete_all="--complete-missing-all" ;;
        u) update="" ;;
        r) remove_explicit="--remove-explicit-failures" ;;
        i) remove_implicit="--remove-implicit-failures" ;;
        \?)
            echo "Error: -${OPTARG} requires an argument."
            exit_abnormal
            ;;
        :)
            echo "Error: -${OPTARG} requires an argument."
            exit_abnormal
            ;;
        *)
            exit_abnormal
            ;;
    esac
done
echo "Starting Compilation with (${trials} min /${max_trials} max) trials"
mkdir -p "${dstbase}"

function update_dst_file(){
    local name=${1}
    local year=${1}
    local trials=${2}
    local src="${srcbase}/${name}"
    local dst_folder="${dstbase}/${name}"
    local dst_file="${dst_folder}/${name}.csv"
    local newpath="${dst_folder}"
    local statsfolder="${dst_folder}stats"
    mkdir -p "${dst_folder}"
    mkdir -p "${statsfolder}"
    echo " =============== Run for ${name} ==================== "
    # python src/helpers/correct_timelimit.py "${src}" "${src}"
    # python src/helpers/clean.py "${src}" --verbose
    # python src/helpers/remove_extra.py "${src}" --output="${dst_file}" --max-trials="${trials}" --remove-failures
    if [ "${name}" == "final" ]; then
        echo "Separating main files"
        python src/helpers/separate_years.py "${dst_file}" --refresh ${verbose};
        python src/helpers/check_failures.py "${dst_file}" ${complete_some} ${complete_all} ${complete_tau} --stats-folder="${statsfolder}" --completion-folder="${newpath}" ${verbose} --fast ${remove_implicit} ${remove_explicit} ${unchecked_strategies}
        echo "Separating completion files"
        for ext in TAU0 AOr AOt; do
            python src/helpers/separate_years.py "${dst_folder}/${name}_complete_all_${ext}.csv" --no-refresh ${verbose};
            python src/helpers/separate_years.py "${dst_folder}/${name}_complete_some_${ext}.csv" --no-refresh ${verbose};
        done
        for y in 0 2010 2011 2012 2013 2015 2016 2017 2018 2019 2020 2021 2022; do
            ybase="${dst_folder}/y${y}"
            mkdir -p "${ybase}"
            mv "${dst_folder}/${name}${y}.csv" "${ybase}"
            for ext in TAU0 AOr AOt; do
                mv "${dst_folder}/final_complete_all_${ext}${y}.csv" "${ybase}"
                mv "${dst_folder}/final_complete_some_${ext}${y}.csv" "${ybase}"
            done
        done
    else
        name="y${year}"
        python src/helpers/copy_runs.py "${dstbase}/final/final.csv" "${dst_file}" --tau=WARNegotiator --tau=CABNegotiator ${verbose} --year=${year}
        python src/helpers/check_failures.py "${dst_file}" ${complete_some} ${complete_all} ${complete_tau} --stats-folder="${statsfolder}" --completion-folder="${newpath}" ${verbose} --fast ${remove_explicit} ${remove_implicit} --year=${year} ${unchecked_strategies}
    fi
}

function compile_dst_file(){
    local year=${1}
    local name=${1}
    local trials=${2}
    if [ "${name}" != "final" ]; then
        name="y${1}"
    fi
    local src="${srcbase}/${name}"
    local dst_folder="${dstbase}/${name}"
    local dst_file="${dst_folder}/${name}.csv"
    local newpath="${dst_folder}"
    local statsfolder="${dst_folder}stats"
    mkdir -p "${dst_folder}"
    mkdir -p "${statsfolder}"
    echo " =============== Run for ${name} ==================== "
    echo " ---- Correcting time limit -----"
    python src/helpers/correct_timelimit.py "${src}" "${src}" ${verbose}
    if [ "${name}" == "final" ]; then
        echo " ---- Cleaning -----"
        python src/helpers/clean.py "${src}"  --outcomelimit="${outcomelimit}" ${remove_explicit} ${remove_implicit} ${excluded_domains} ${verbose} ${verbose} --min-trials="${trials}"  --max-trials="${max_trials}"  ${remove_implicit} ${remove_explicit} ${remove_negative}
        echo " ---- Removing Extra -----"
        python src/helpers/remove_extra.py "${src}" --output="${dst_file}" --min-trials="${trials}"  --max-trials="${max_trials}" --outcomelimit="${outcomelimit}" ${verbose}
        # python src/helpers/remove_extra.py "${dst_file}" --output="${dst_file}" --max-trials="${max_trials}"  --remove-failures --remove-implicit-failures --no-verbose
        echo " ---- Separating main files ----"
        python src/helpers/separate_years.py "${dst_file}" --refresh ${verbose};
        echo " ---- Checking Failures ----"
        python src/helpers/check_failures.py "${dst_file}" --no-complete-missing-some --no-complete-missing-all --no-complete-tau --no-complete-proposed --stats-folder="${statsfolder}" --completion-folder="${newpath}" ${verbose} --fast --n-trials="${max_trials}" --min-n-trials="${max_trials}" ${unchecked_strategies}
        echo " --- Separating completion files --- "
        for ext in TAU0 AOr AOt; do
            python src/helpers/separate_years.py "${dst_folder}/${name}_complete_all_${ext}.csv" --no-refresh ${verbose};
            python src/helpers/separate_years.py "${dst_folder}/${name}_complete_some_${ext}.csv" --no-refresh ${verbose};
        done
        for y in 0 2010 2011 2012 2013 2015 2016 2017 2018 2019 2020 2021 2022; do
            ybase="${dst_folder}/y${y}"
            mkdir -p "${ybase}"
            mv "${dst_folder}/${name}${y}.csv" "${ybase}"
            for ext in TAU0 AOr AOt; do
                mv "${dst_folder}/final_complete_all_${ext}${y}.csv" "${ybase}"
                mv "${dst_folder}/final_complete_some_${ext}${y}.csv" "${ybase}"
            done
        done
    else
        echo " ---- Cleaning -----"
        python src/helpers/clean.py "${src}"  --outcomelimit="${outcomelimit}" ${remove_explicit} ${remove_implicit} ${excluded_domains} ${verbose} ${verbose} --min-trials="${trials}"  --max-trials="${max_trials}"  ${remove_implicit} ${remove_explicit} ${remove_negative}
        echo " ---- Removing Extra -----"
        python src/helpers/remove_extra.py "${src}" --output="${dst_file}" --min-trials="${trials}"  --max-trials="${max_trials}" --outcomelimit="${outcomelimit}" ${verbose}
        # python src/helpers/remove_extra.py "${dst_file}" --output="${dst_file}" --max-trials="${trials}"  --remove-failures --remove-implicit-failures --no-verbose
        # echo " ---- Copying runs -----"
        # python src/helpers/copy_runs.py "${dstbase}/final/final.csv" "${dst_file}" --tau=WARNegotiator --tau=CABNegotiator ${verbose} --year=${year}
        if [[ -n "$remove_nonfinalists" ]]; then
            echo "--removing nonfinalists--"
            python src/helpers/remove_extra_agents.py "${dst_file}" ${year} --remove-non-finalists --no-remove-non-genius10
        fi
        if [[ -n "$remove_nongenius10" ]]; then
            echo "--removing non Genius10--"
            python src/helpers/remove_extra_agents.py "${dst_file}" ${year} --no-remove-non-finalists --remove-non-genius10
        fi
        echo " ---- Checking Failures -----"
        python src/helpers/check_failures.py "${dst_file}" --no-complete-missing-some --no-complete-missing-all --no-complete-tau --no-complete-proposed --stats-folder="${statsfolder}" --completion-folder="${newpath}" ${verbose} --fast --n-trials="${max_trials}" --min-n-trials="${max_trials}" --year=${year} --original-runs-only ${unchecked_strategies}
    fi
}
echo "-- combining local results with server results --"
combineresults.sh ${srcbase}
echo "--correcting agent names--"
python src/helpers/correct_names.py "${srcbase}" ${verbose}
if [[ "${update}" == "yes" ]]; then
    rm -rf "${dstbase}"
    compile_dst_file final "${trials}"
    for name in $peryear; do
        compile_dst_file "${name}" "${trials}"
    done
else
    fd ".*complete.*.csv" "${dstbase}" -X rm
    fd "missing.*.csv" "${dstbase}" -X rm
    fd "runs.*.txt" "${dstbase}" -X rm
    update_dst_file final "${trials}"
    for name in $peryear; do
        update_dst_file "${name}" "${trials}"
    done
fi
