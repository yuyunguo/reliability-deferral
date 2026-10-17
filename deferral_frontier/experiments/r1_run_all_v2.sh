#!/bin/sh
# Round-1 composite-v2 full rerun: pipelines -> downstream -> verification -> split-seed sensitivity.
# Usage (from project root): sh deferral_frontier/experiments/r1_run_all_v2.sh
P=.venv/bin/python
M=deferral_frontier.experiments
: > logs/v2_full_status.log
st() { echo "$1" >> logs/v2_full_status.log; }
for t in sepsis mortality vasopressor sepsis3; do $P -m $M.run_deferral $t > logs/v2_mimic4_$t.log 2>&1 || st "FAILED mimic4 $t"; done
for t in sepsis mortality vasopressor; do $P -m $M.run_mimic3_validation $t > logs/v2_mimic3_$t.log 2>&1 || st "FAILED mimic3 $t"; done
for sp in patient hospital; do
  for t in sepsis mortality vasopressor; do
    $P -m $M.run_eicu_validation $t --split $sp > logs/v2_eicu_${sp}_$t.log 2>&1 || st "FAILED eicu $sp $t"
  done
done
st PIPELINES_DONE
sh deferral_frontier/experiments/r1_run_downstream.sh
st DOWNSTREAM_DONE
$P -m $M.r1_verify_pipeline > logs/v2_verify.log 2>&1 || st "FAILED verify"
$P -m $M.r1_split_seed_sensitivity --sources MIMIC-IV MIMIC-III eICU eICU-H > logs/v2_split_seed.log 2>&1 || st "FAILED split seed"
st DONE
