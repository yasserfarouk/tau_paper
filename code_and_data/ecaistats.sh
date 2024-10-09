#!/usr/bin/env bash
# generates all figures for paper Generalized Bargaining Protocols
python src/ecaistats.py --dataset=all --cabwar --name-stem=everything --rename-cab --remove-cab-vs-war --statistics=mean --statistics=std --no-relative
python src/ecaistats.py --dataset=all --cabwar --name-stem=everything_relative --rename-cab --remove-cab-vs-war --statistics=mean --statistics std --relative
# python src/ecaistats.py --dataset=all --no-cabwar --name-stem=adapter --rename-cab --statistics=mean
# python src/ecaistats.py --dataset=all --no-cabwar --name-stem=adapter_relative --rename-cab --statistics=mean --relative
# python src/ecaistats.py --dataset=all --no-cabwar --name-stem=adapter --rename-cab
# scalability.sh
# cp -r figs/ecai "$DST"
