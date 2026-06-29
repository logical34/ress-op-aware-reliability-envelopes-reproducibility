"""Top-level reproducibility runner.

Run:
    python simulation.py

This delegates to scripts/simulate_reliability.py and scripts/cmapss_case_study.py.
It regenerates the controlled Case A/B outputs and the C-MAPSS public-benchmark
Case C summaries used by the manuscript.
"""

from scripts.simulate_reliability import main as run_controlled_cases
from scripts.cmapss_case_study import main as run_cmapss_case
from scripts.generate_manuscript_tables import main as generate_manuscript_tables


if __name__ == "__main__":
    run_controlled_cases()
    run_cmapss_case()
    generate_manuscript_tables()
