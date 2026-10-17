"""Tests for the redesigned eICU cohort and labels (round 1)."""
import pandas as pd

from deferral_frontier.experiments.extract_eicu_r1 import eicu_sepsis_flag, select_eicu_cohort


def test_sepsis_flag_keeps_sepsis_category_and_drops_sirs_and_local_infection():
    s = pd.Series([
        "cardiovascular|shock / hypotension|sepsis",
        "infectious diseases|systemic/other infections|sepsis|severe",
        "cardiovascular|shock / hypotension|septic shock|cultures pending",
        "infectious diseases|systemic/other infections|signs and symptoms of sepsis (SIRS)",
        "renal|disorder of kidney|acute renal failure|due to sepsis",
        "infectious diseases|skin, bone and joint infections|septic arthritis",
    ])
    assert eicu_sepsis_flag(s).tolist() == [True, True, True, False, False, False]


def test_select_eicu_cohort_first_eligible_unit_stay_per_hospitalization():
    stays = pd.DataFrame({
        "stay_id": [1, 2, 3, 4, 5, 6],
        "hosp_stay_id": [10, 10, 20, 30, 40, 50],
        "unitvisitnumber": [2, 1, 1, 1, 1, 1],
        "unitdischargeoffset": [3000, 2000, 1000, 5000, 5000, 5000],   # minutes
        "age": [60, 60, 50, 17, 70, 70],
        "hospitaldischargestatus": ["Alive", "Alive", "Alive", "Alive", "", "Expired"],
    })
    out = select_eicu_cohort(stays)
    assert list(out.index) == [2, 6]          # index = stay_id
    assert out.loc[6, "mortality"] == 1 and out.loc[2, "mortality"] == 0
