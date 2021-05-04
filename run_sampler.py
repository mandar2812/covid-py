### −∗− mode : python ; −∗−
# @file run_sampler.py
# @author Mandar Chandorkar
######################################################

from pathlib import Path
import argparse
from typing import Any, Dict, List, Optional
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from cycler import cycler
import prettyprinter
from prettyprinter import cpprint
import pymc3 as pm

import models
import data_utils

prettyprinter.install_extras()


def load_data(state: str) -> pd.DataFrame:
    """Load data for a state."""
    return data_utils.load_covid_data_india().get(state)

# def sampling_exp(state: str, start_date: datetime.datetime):
#     """Perform the sampling experiment."""
#     data: pd.DataFrame = load_data(state)
#     with pm.Model() as model:
#         # Exposure factor
#         beta = pm.Uniform("beta", lower=0.7, upper=1.5) # 1.2
#         # Sympotmatic vs Asymptomatic
#         pa = pm.Uniform("pa", lower=0.2, upper=0.6)
#         r_b = 0.2
#         # Recovery rate
#         mu = 0.02
#         eps = 1.0 / 14
#         # Re-infection rate
#         rho = 1 / 10000 #(8 * 30)
#         # Death rate
#         p_d = 0.02

