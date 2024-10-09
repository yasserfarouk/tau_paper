# Supplementary Materials for "Improving Automated Negotiation Outcomes using the Tentative Agreement Unique Offers Protocol" (Submitted to JAAMAS in 2024)

This folder contains code and data for running all the experiments
described in the paper and the technical appendix.

## Reproducibility and Public Availability

- This guide shows how to reproduce every result in the paper.
- The data is already publicly available in the GENIUS platform.
- All the contents of our supplementary materials will be made publicly
  available on Github in case of acceptance. Moreover, we will contribute
  our algorithms back the negotiation platform we use (NegMAS) as a pull
  request and if accepted will be available directly from the platform
  (which is community driven) in the future.

## What is included?

- **`appendix.pdf`** The technical appendix
- **`code_and_data`** Raw data and implementation of algorithms.
  - **`scenarios`** All the negotiation scenarios used in the paper.
  - **`serverclean`** Example of the experimental results for year 2010
    (all strategies). The full dataset is too large to include but it
    consists of more files of exactly the same format and can be
    generated as described later in this guide.
  - **`src`** The source code used for implementing the protocol, adapter
    and all evaluation experiments.
    - **`mechanisms.py`** A full implementation of the TAU protocol
    - **`adapters/tau.py`** Full implementation of the proposed adaptation algorithm
  - **`main_experiment.sh`** Runs the main experiment reported in the paper
    and detailed in the technical appendix.
  - **`peryearall.sh`** Runs the remaining 8 experiments reported in the paper by
    passing the years "2010 2011 2012 2013 2015 2016 2017 2018"
  - **`ijcaifigs.sh`** Generates the main tables and figures in the paper and
    appendix (and much more). Find them at `./figs` after running the script.
    Must be run after completion of `main_experiment.sh` and `peryear.sh`
  - **`detailedfigs.sh`** Generates detailed figures used only in the appendix
  - **`ijcaigames.sh`** Generates game theoretic results including equilibria
    calculation and replicator dynamics
  - **`scalability.sh`** Generates scalability results

## Software Requirements

- Python `3.11` (tested on MacOS with Python 3.11.4).
- Java `18` (tested on MacOS with OpenJDK 18 2022-03-22 build 18+36-2087).
  - only needed to run state-of-the-art negotiators (and Nice Tit for Tat)
- negmas `0.10.12`. Please use the version pinned in the requirements.txt file in the `code_and_data` folder.
- genius-bridge `v0.4.13` (installed when following the instructions in the following section).
  - only needed to run state-of-the-art negotiators (and Nice Tit for Tat)

## Installing Requirements

Please use your platform's preferred method to install Python 3.11+ and Java 18+.

To install other requirements run the following command within the `code` folder:

```bash
pip install -r code_and_data/requirement_pinned.txt
negmas genius-setup
```

Note that the later command will download the `negmas-genius` bridge and install in as:

```
$HOME/negmas/files/geniusbridge.jar
```

We assume that this is run within a virtual environment (as always recommended).

## Running Experiments

_Assumes that you installed requirements_

To run the experiment reported in the paper and generate all results:

```bash
> cd code_and_data
> python src/helpers/prepare.py scenarios
> python src/make_finalist_datasets.py
> main_experiment.sh
> for y in 10 11 12 13 15 16 17 18; do python peryearall.sh 20$y; done
> ijcaifigs.sh; ijcaigames.sh; ijcaistats.sh; detailefigs.sh; scalability.sh
```

This will take several weeks to complete as it runs around 380K negotiations some of them taking hours.
You can pass --limitoutcomes=1000 for example to limit the run to scenarios with no more than a thousand outcomes.
See the documentation of main_experiment.sh about how to pass this parameter.
For peryearall.sh, you can pass it directly on the command line.
