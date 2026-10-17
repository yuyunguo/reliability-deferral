"""Extract demographics from MIMIC-IV for fairness audit.

Reads test artifact hadm_ids and queries the MIMIC-IV PostgreSQL database
for race, gender, and age. Saves to data/mimic4/{task}_demographics.parquet.

Usage:
    source deferral_frontier/manuscript/mimiciii_config.sh  # for DB credentials
    python -m deferral_frontier.experiments.extract_demographics sepsis
    python -m deferral_frontier.experiments.extract_demographics mortality
    python -m deferral_frontier.experiments.extract_demographics vasopressor
"""
import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path("artifacts")
DATA_DIR = Path("data/mimic4")

logger = logging.getLogger(__name__)

RACE_MAPPING = """
    CASE
        WHEN a.race LIKE '%%WHITE%%' THEN 'White'
        WHEN a.race LIKE '%%BLACK%%' THEN 'Black'
        WHEN a.race LIKE '%%HISPANIC%%' OR a.race LIKE '%%LATINO%%' THEN 'Hispanic'
        WHEN a.race LIKE '%%ASIAN%%' THEN 'Asian'
        WHEN a.race = 'UNKNOWN' OR a.race = 'UNABLE TO OBTAIN' THEN 'Unknown'
        ELSE 'Other'
    END
"""


def extract_demographics(task: str) -> None:
    import psycopg2

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    test_path = ARTIFACTS_DIR / f"{task}_test_artifacts.parquet"
    if not test_path.exists():
        logger.error("Run run_deferral.py first to generate artifacts.")
        return

    test = pd.read_parquet(test_path)
    hadm_ids = test["hadm_id"].tolist()
    logger.info("Extracting demographics for %d test patients", len(hadm_ids))

    host = os.environ.get("MIMICIV_DB_HOST", "localhost")
    port = os.environ.get("MIMICIV_DB_PORT", "5432")
    dbname = os.environ.get("MIMICIV_DB_NAME", "mimiciv")
    user = os.environ.get("MIMICIV_DB_USER", "postgres")
    password = os.environ.get("MIMICIV_DB_PASSWORD", "")
    schema = os.environ.get("MIMICIV_DB_SCHEMA", "mimic_core")

    conn = psycopg2.connect(host=host, port=port, dbname=dbname,
                            user=user, password=password)
    try:
        query = f"""
            SELECT a.hadm_id,
                   p.gender,
                   p.anchor_age,
                   {RACE_MAPPING} AS race_group
            FROM {schema}.admissions a
            JOIN {schema}.patients p ON a.subject_id = p.subject_id
            WHERE a.hadm_id = ANY(%s)
        """
        demo = pd.read_sql(query, conn, params=(hadm_ids,))
        demo = demo.set_index("hadm_id")
        logger.info("Extracted %d demographic records", len(demo))
        logger.info("Race distribution:\n%s", demo["race_group"].value_counts().to_string())
        logger.info("Gender distribution:\n%s", demo["gender"].value_counts().to_string())

        out_path = DATA_DIR / f"{task}_demographics.parquet"
        demo.to_parquet(out_path)
        logger.info("Saved to %s", out_path)
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, nargs="?", default="sepsis")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    extract_demographics(args.task)
