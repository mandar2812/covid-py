"""Utilities for loading the covid data sets.

@author Mandar Chandorkar.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import pandas as pd


DATA_DIR: Path = Path.cwd() / ".data"


def load_covid_data_india() -> Dict[str, pd.DataFrame]:
    """Load statewise covid infection data."""
    covid_data_india = pd.read_csv(DATA_DIR / "india" / "covid_19_india.csv")
    covid_data_india["Date"] = pd.to_datetime(covid_data_india["Date"])
    return {
        k: v
        for k, v in covid_data_india.groupby("State/UnionTerritory")
    }


def load_vaccination_data_india() -> Dict[str, pd.DataFrame]:
    """Load statewise covid vaccination data."""
    covid_data_vacc_india = pd.read_csv(DATA_DIR / "india" / "covid_vaccine_statewise.csv")
    covid_data_vacc_india["Updated On"] = pd.to_datetime(covid_data_vacc_india["Updated On"])
    return {
        k: v
        for k, v in covid_data_vacc_india.groupby("State")
    }


def plot_covid_data_state(
        covid_df: Dict[str, pd.DataFrame],
        state: str,
        population: int,
        start_date=None):
    """Plot the covid numbers for a state."""
    if start_date is not None:
        case_data = covid_df[state].loc[covid_df[state].Date > start_date]
    else:
        case_data = covid_df[state]

    projected_case_data = pd.DataFrame(
        {
            "Date": case_data.Date,
            "rest": population - (case_data["Confirmed"] - case_data["Cured"]),
            "infected": case_data["Confirmed"] - (case_data["Cured"] + case_data["Deaths"]),
            "dead": case_data["Deaths"]
        }
    )
    ax = projected_case_data.plot(x="Date", y=["infected", "dead"], figsize=(12, 8), title=state)
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='-', linewidth=0.25, axis='both', alpha=0.85, color="black")
    ax.grid(True, which='minor', linestyle='-.', linewidth=0.25, axis='both', alpha=0.7)
    return ax
