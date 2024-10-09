#!/usr/bin/env bash
# generates all figures for paper Generalized Bargaining Protocols

python src/ecaifigs.py --dataset=all --name-stem=everything --rename-cab --remove-cab-vs-war
python src/ecaistats.py --dataset=all --cabwar --name-stem=everything --rename-cab --remove-cab-vs-war
scalability.sh
# python src/ecaistats.py --dataset=all --no-cabwar --name-stem=adapter --rename-cab
# python src/ecaifigs.py --dataset="all" --no-cabwar --name-stem="adapter" --rename-cab
# python src/ecaifigs.py --dataset="all" --remove-aop --name-stem="tau_only" --rename-cab
# python src/ecaifigs.py --dataset="all" --no-adapters --name-stem="tau_native" --rename-cab
# python src/ecaifigs.py --dataset="all" --no-adapters --tau-pure --name-stem="tau_pure" --rename-cab
