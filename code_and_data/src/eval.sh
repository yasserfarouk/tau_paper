#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
usage() {                                 # Function: Print a help message.
	echo "Usage: $("basename $0") [-i] [-s] [-q] [-f source] [-y year] [-n precision] [-t trials] [-o] [-c] [-v] [-b] [-m] [-d path] [-p] [-P] [-z] [-D] [-A] [-T] [-E] [-R] [-O]" 1>&2
    echo "   -f: Source which can be a file or a directory"
    echo "   -y: Year to execute scenarios from. Use 0 to get combined results of all years and -1 to get results for each individual year as well as combined results"
    echo "   -n: Precision in printing to latex"
    echo "   -b: Bilateral only"
    echo "   -m: Multilateral only"
    echo "   -s: Remove mechanism name to shorten names in legends"
    echo "   -i: If given, second agent's utils will be ignored when calculating agent stats"
    echo "   -v: If given, the figures will be shown, otherwise just saved directly"
    echo "   -d: Path to a folder to which we copy all results"
    echo "   -q: Quicker: No individual figs, no stats, not tables, no ttests"
    echo "   -Q: Quick No no stats, not tables, no ttests"
    echo "   -t: Maximum number of trials to use for each condition"
    echo "   -c: If given, data will be cleaned first"
    echo "   -o: override all earlier runs. Will delete all older runs"
	echo "   -O: if given only original data will be used not data with controlled resreved value"
    echo "   -p: If given TAU will be pure"
    echo "   -P: If given all considered runs will be pure"
    echo "   -z: If given adapted runs will be removed"
	echo "   -v: If given TAU variants (e.g. WAN, CAR, ...) will be removed"
    echo "   -D: If given calculates only designer view results"
    echo "   -A: If given calculates only agent view results"
    echo "   -T: excluding timing in combined figures"
    echo "   -R: remove incomplete domains"
	echo "   -E: if given we will remove implicit failures resulting from exceptions from consideration"
}
exit_abnormal() {
  usage
  exit 1
}
topn=3
year=0
override=""
clean=""
base="results"
precision=2
trials=3
display="--no-show"
bilateral="--bilateral"
multilateral="--multilateral"
ignoresecond="--no-ignore-second"
dst=""
quick="--figs-individual --figs-line --figs-bar --stats --ttests"
shorten=""
adapted="--adapted"
variants="--tau-variants"
pure="--no-pure-only"
taupure="--no-pure-tau"
included=""
excluded=""
view="--agent-view --designer-view"
excludetiming="--no-exclude-timing"
noexceptions="--no-remove-implicit-failures"
removeincomplete="--no-remove-incomplete-domain"
original="--original"
controlled="--controlled"
lbl="impure"
resultspath="$(dirname $(dirname "${SCRIPT_DIR}"))"
dominatedonly="--no-filter-dominated"
while getopts "COETzPVpQqimbsocDRAvFr:k:n:t:f:y:d:" options; do
  case "${options}" in
    F) dominatedonly="--filter-dominated"  ;;
    Q) quick="--no-stats --no-ttests --no-figs-line --figs-bar"  ;;
    q) quick="--no-stats --no-ttests --no-figs-individual --no-figs-line --figs-bar"  ;;
    O) controlled="--no-controlled"  ;;
    C) original="--no-original"  ;;
    E) noexceptions="--remove-implicit-failures"  ;;
    R) removeincomplete="--remove-incomplete-domain" ;;
    T) excludetiming="--exclude-timing"  ;;
    i) ignoresecond="--ignore-second"  ;;
    m) bilateral="--no-bilateral"  ;;
    b) multilateral="--no-multilateral"  ;;
    s) shorten="--shorten-names"  ;;
    o) override="yes" ;;
    c) clean="yes" ;;
    v) display="--show" ;;
    p) taupure="--pure-tau" ;;
    P) pure="--pure-only"; lbl="pure" ;;
    z) adapted="--no-adapted" ;;
    V) variants="--no-tau-variants" ;;
    D) view="--no-agent-view --designer-view" ;;
    A) view="--agent-view --no-designer-view" ;;
    r) resultspath="${OPTARG}" ;;
    n) precision=${OPTARG} ;;
    t) trials=${OPTARG} ;;
    f) base=${OPTARG} ;;
    y) year=${OPTARG} ;;
    d) dst=${OPTARG} ;;
    k) topn=${OPTARG} ;;
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
echo "Reading from ${base}"
echo "Saving results to ${resultspath}"
if [[ -n ${clean} ]]; then
	python clean.py "${base}" --max-trials="${trials}"
fi
rm -f tables/*.log
rm -f tables/*.aux
if [[ -n ${override} ]]; then
	rm -rf tables/*
	rm -rf figs/*
	rm -rf stats/*
fi
# included="--include=MiCRONegotiator --include=AspirationNegotiator --include=Atlas3 --include=CABNegotiator --include=WARNegotiator"
# excluded="--exclude=WANNegotiator --exclude=WABNegotiator --exclude=CANNegotiator --exclude=CARNegotiator --tau-exclude=AspirationNegotiator --tau-exclude=MiCRONegotiator"

function generate_all(){
	year=$1
	echo "********** Working on year ${year} from ${base} ********"
	# python ${SCRIPT_DIR}/makefigs.py --files="${base}" --max-trials="${trials}" --rational-only --precision="${precision}" --topn="${topn}" --output="${base}/${lbl}/fig" --year="${year}" ${display} ${excluded} ${included} --max-trials=0 ${bilateral} ${multilateral} ${ignoresecond} ${shorten} ${pure} ${taupure} ${adapted} ${variants} ${view} ${excludetiming} ${noexceptions} ${original} ${controlled} ${removeincomplete} --results-path="${resultspath}" --no-stats --no-ttests --filter-dominated
	echo ${SCRIPT_DIR}/makefigs.py --files="${base}" --min-trials="${trials}" --max-trials="${trials}" --rational-only --precision="${precision}" --topn="${topn}" --output="${base}/${lbl}/fig" --year="${year}" ${display} ${excluded} ${included} --max-trials=${trials} ${bilateral} ${multilateral} ${ignoresecond} ${shorten} ${pure} ${taupure} ${adapted} ${variants} ${view} ${excludetiming} ${noexceptions} ${original} ${controlled} ${removeincomplete} --results-path="${resultspath}" ${quick} ${dominatedonly}
	python ${SCRIPT_DIR}/makefigs.py --files="${base}" --min-trials="${trials}" --max-trials="${trials}" --rational-only --precision="${precision}" --topn="${topn}" --output="${base}/${lbl}/fig" --year="${year}" ${display} ${excluded} ${included} --max-trials="${trials}" ${bilateral} ${multilateral} ${ignoresecond} ${shorten} ${pure} ${taupure} ${adapted} ${variants} ${view} ${excludetiming} ${noexceptions} ${original} ${controlled} ${removeincomplete} --results-path="${resultspath}" ${quick} ${dominatedonly}
}

function copyall(){
	if [[ -n ${dst} ]]; then
		# rm -rf "${dst}"
		mkdir -p "${dst}"
		cp -rf figs/* "${dst}"
	fi
}

if [ "${year}" -lt 0 ]; then
	if [[ -n ${bilateral} && -n ${multilateral} ]]; then
		years="0 2010 2022 2011 2012 2013 2015 2016"
	elif [[ -n ${bilateral} ]]; then
		years="0 2010 2022 2011 2012 2013"
	elif [[ -n ${bilateral} ]]; then
		years="0 2015 2016"
	else
		years="0"
	fi
	for y in ${years}; do
		generate_all "${y}"
		copyall
	done
else
	generate_all "${year}"
	copyall
fi
rm -f tables/*.log
rm -f tables/*.aux
