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
