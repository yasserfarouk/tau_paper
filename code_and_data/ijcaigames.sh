#!/bin/bash

for s in mean; do
    for game in everything taunative tau; do
        python src/game.py --dataset=all --stat=${s} --name-stem=${s} --game=${game} --solver=any --no-overwrite-data-files --correct-reading-errors --rename-cab --resolution=1 --generations=10_000 --max-cores=-1 --no-unknown-order --name-stem="asymmetric" $@
        python src/game.py --dataset=all --stat=${s} --name-stem=${s} --game=${game} --solver=any --no-overwrite-data-files --correct-reading-errors --rename-cab --resolution=1 --generations=10_000 --max-cores=-1 --unknown-order --name-stem="symmetric" $@
    done
done
