#!/usr/bin/env sh

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
usage() {                                 # Function: Print a help message.
    echo "Usage: $("basename $0") [-n outcome-limit] [-S] [-r] [-l time-limit] [-t trials] [-T max-allowed-time] [-s] [-o] [-v] [-d] [-y year] [-c cores] [-D] [-P] [-A] [-G] [-R] [-Y] [-o] [-V]" 1>&2
    echo "   -n: maximum outcome space size to use"
    echo "   -M: n. groups of files"
    echo "   -m: my group of files"
    echo "   -o: Order by # outcome (See -V)"
    echo "   -V: Reverse the order (larger to smaller)"
    echo "   -l: time limit for AOP runs"
    echo "   -t: N trials"
    echo "   -r: run negotiations in random order instead of from smaller to larger outcome spaces"
    echo "   -T: Maximum allowed time to force for any negotiation"
    echo "   -y: Year to execute scenarios from"
    echo "   -Y: All years: 2010,11,12,13,15,16,17,18,19,20,21,22"
    echo "   -s: serial execution"
    echo "   -S: fast start (slow run)"
    echo "   -v: verbose"
    echo "   -d: debug"
    echo "   -D: Dummy (no run)"
    echo "   -c: number of cores to use"
    echo "   -o: override all earlier runs. Will delete all older runs"
    echo "   -P: Do not prepare"
    echo "   -f: base folder (defaults to scenarios)"
    echo "   -z: Number of steps after which we consider an exception from an agent the same as ending the negotiation. Pass <= 0 to ignore"
    echo "   -Z: fraction (0-1) of steps after which we consider an exception from an agent the same as ending the negotiation. Pass <= 0 to ignore"
}
exit_abnormal() {
    usage
    exit 1
}
year=0
timelimit=180
maxallowedtime=0
outcomelimit=40000000
trials=1
serial="--no-serial"
override="--no-override"
verbose=0
debug="--no-debug"
breakonexception="--no-break-on-exception"
norun="--no-norun"
cores=48
order="--order"
reversed="--no-reversed"
prepare="yes"
allyears=""
faststart=""
nG=-1
group=-1
minoutcomes=0
basefolder="scenarios"
reserved_on_failure="--no-reserved-value-on-failure"
ntosavefailure=0
rtosavefailure=0
while getopts "Z:z:FUBSVYrDsdoPf:m:v:c:T:t:l:y:n:M:N:" options; do
    case "${options}" in
        c) cores=${OPTARG} ;;
        M) nG=${OPTARG} ;;
        m) group=${OPTARG} ;;
        f) basefolder=${OPTARG} ;;
        l) timelimit=${OPTARG} ;;
        t) trials=${OPTARG} ;;
        T) maxallowedtime=${OPTARG} ;;
        y) year=${OPTARG} ;;
        n) outcomelimit=${OPTARG} ;;
        N) minoutcomes=${OPTARG} ;;
        v) verbose=${OPTARG} ;;
        z) ntosavefailure=${OPTARG} ;;
        Z) rtosavefailure=${OPTARG} ;;
        P) prepare="" ;;
        D) norun="--norun" ;;
        F) reserved_on_failure="--reserved-value-on-failure" ;;
        S) faststart="--fast-start" ;;
        Y) allyears="yes" ;;
        r) order="--no-order" ;;
        V) reversed="--reversed" ;;
        s) serial="--serial"; cores=-1 ;;
        d) debug="--debug" ;;
        B) breakonexception="--break-on-exception" ;;
        o) override="--override" ;;
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

if  [ -n "$allyears" ]; then
    year=0
fi
base="${SCRIPT_DIR}/${basefolder}"
donefolder="serverclean"
if  [ -n "$prepare" ]; then
    # Prepare domains if not already prepared
    if [ ${year} -eq  0 ]; then
        python src/helpers/prepare.py --base="${base}" --outcomelimit="${outcomelimit}" --minoutcomes="${minoutcomes}" --no-override
        donefolder="serverclean/final.csv"
    else
        python src/helpers/prepare.py --base="${base}/y${year}" --outcomelimit="${outcomelimit}" --minoutcomes="${minoutcomes}" --no-override
        donefolder="serverclean/y${year}/y${year}.csv"
    fi
fi
# Running with everything
python src/experiment.py --base=${base} ${faststart} --allcombinations --unique-file-name --save-agreement --f0=1 --f1=1 --f1=0.1 --f1=0.5 --f1=0.9 --outcomelimit="${outcomelimit}" --minoutcomes=${minoutcomes} --year="${year}" --timelimit=${timelimit} --rounds=1 --trials="${trials}" "${serial}" "${override}" --verbose="${verbose}" "${debug}"  "${breakonexception}" --max-cores="${cores}" --title="1. All together AOP, TAU, Adapters ${year}" ${norun} --max-allowed-time=${maxallowedtime} ${reserved_on_failure} --done=${donefolder} --done=results ${order} ${reversed} --ngroups=${nG} --group=${group} --steps-to-register-failure=${ntosavefailure} --relative-steps-to-register-failure=${rtosavefailure} --add-cab-war
