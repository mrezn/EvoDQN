from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .utils import ensure_dir


def save_latex_table(path: Path, caption: str, label: str, df: pd.DataFrame) -> None:
    latex = df.to_latex(index=False, caption=caption, label=label, escape=True)
    path.write_text(latex, encoding="utf-8")


def export_tables(config: dict, output_dirs: Dict[str, Path]) -> Dict[str, Path]:
    tables_dir = ensure_dir(output_dirs["tables"])

    strategies_rows = []
    for strategy in config.get("env", {}).get("strategies", {}).get("G_A", []):
        strategies_rows.append(
            {
                "Role": "Attacker",
                "Strategy": strategy,
                "Mechanism": "Adaptive manoeuvre across WF attack graph",
                "Impact": "Increases SAL via stealthy pivoting",
                "Key knobs": "p_i capability",
                "Uncertainty": "Reward scale T^A and transition ambiguity ??",
            }
        )
    for strategy in config.get("env", {}).get("strategies", {}).get("G_D", []):
        strategies_rows.append(
            {
                "Role": "Defender",
                "Strategy": strategy,
                "Mechanism": "reconfiguration of edge resources",
                "Impact": "Mitigates SAP loss and improves assurance",
                "Key knobs": "c_* rate a_r, SQ",
                "Uncertainty": "Reward scale TD, bounded rational policy",
            }
        )
    strategies_df = pd.DataFrame(strategies_rows)
    strategy_csv = tables_dir / "table_strategies.csv"
    strategies_df.to_csv(strategy_csv, index=False)
    strategy_tex = tables_dir / "table_strategies.tex"
    save_latex_table(strategy_tex, "Attacker and defender strategies", "tab:strategies", strategies_df)

    vulnerabilities_df = pd.DataFrame(
        [
            {
                "Component": "Sensor",
                "Vulnerability": "Firmware overflow and spoofed telemetry",
                "Mitigation": "MTD schedule, integrity attestation",
            },
            {
                "Component": "Edge host",
                "Vulnerability": "Credential replay and lateral movement",
                "Mitigation": "IP/port hopping, rate limiting",
            },
            {
                "Component": "Control layer",
                "Vulnerability": "Command injection via stale sessions",
                "Mitigation": "Adaptive isolation, strong detection ?",
            },
            {
                "Component": "Cloud",
                "Vulnerability": "Data exfiltration under weak configs",
                "Mitigation": "Time randomisation, resource hardening",
            },
        ]
    )
    vuln_csv = tables_dir / "table_vulnerabilities.csv"
    vulnerabilities_df.to_csv(vuln_csv, index=False)
    vuln_tex = tables_dir / "table_vulnerabilities.tex"
    save_latex_table(vuln_tex, "Attack surface overview", "tab:vulnerabilities", vulnerabilities_df)

    return {
        "strategies_csv": strategy_csv,
        "strategies_tex": strategy_tex,
        "vulnerabilities_csv": vuln_csv,
        "vulnerabilities_tex": vuln_tex,
    }
