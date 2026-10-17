#!/bin/sh
# Round-1 downstream reruns on regenerated artifacts (plan Task 1.5 + Phase 2).
# Usage: sh deferral_frontier/experiments/r1_run_downstream.sh   (from project root)
PY=.venv/bin/python
M=deferral_frontier.experiments
mkdir -p logs; : > logs/r1_downstream_status.log
run() { name=$1; shift; if $PY -m "$@" > "logs/ds_$name.log" 2>&1; then echo "ok $name" >> logs/r1_downstream_status.log; else echo "FAILED $name" >> logs/r1_downstream_status.log; fi; }
run symmetric_gate      $M.run_symmetric_confidence_gate
run paired_gate_tests   $M.run_paired_gate_tests
run significance        $M.run_significance_tests
run decision_curve      $M.run_decision_curve
for t in sepsis mortality vasopressor; do run conformal_$t $M.run_conformal_baseline $t; done
run alternative_uq      $M.run_alternative_baselines
run d5_latency          $M.run_d5_ablation_latency
run vif                 $M.run_vif_table
run entropy             $M.run_entropy_decomposition
run calibration_diag    $M.run_calibration_diagnostics
run imperfect_review    $M.run_imperfect_review
run fairness_r1         $M.run_fairness_r1
run clinical_scores     $M.run_clinical_score_baseline
run sepsis3_summary     $M.r1_sepsis3_summary
run neural_baselines    $M.run_neural_baselines
run shift               $M.run_shift_experiments
run sym_figures         $M.plot_symmetric_gate_figures
run v2_ablation         $M.r1_v2_ablation
run before_after        $M.r1_before_after
echo DONE >> logs/r1_downstream_status.log
