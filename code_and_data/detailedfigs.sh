#!/usr/bin/env bash
# generates all figures for paper Generalized Bargaining Protocols

python src/detailedfigs.py --dataset=final  --no-scatter --rename-cab $@
