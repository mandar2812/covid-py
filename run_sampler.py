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
from ray import tune
# from ray.tune.suggest import ConcurrencyLimiter
# from ray.tune.schedulers import AsyncHyperBandScheduler
# from ray.tune.suggest.dragonfly import DragonflySearch
import models
import data_utils
prettyprinter.install_extras(exclude=["ipython", "django"])


def load_data(state: str, start_date: datetime.datetime) -> pd.DataFrame:
    """Load data for a state."""
    df = data_utils.load_covid_data_india().get(state)
    return df.loc[df.Date >= start_date].copy()


def tuning_exp(
        state: str,
        population: int,
        start_date: datetime.datetime,
        num_samples: int = 50):
    """Perform the sampling experiment."""
    def _tuning_fn(config: Dict[str, Any]):
        data: pd.DataFrame = load_data(state, start_date)
        simulator = models.seiird(
            beta=config["beta"],
            p_a=config["p_a"],
            r_b=config["r_b"],
            eps=config["eps"],
            mu=config["mu"],
            rho=config["rho"],
            p_d=config["p_d"]
        )

        start_mh = data[data.Date == start_date].iloc[-1]
        I0 = int(start_mh.Confirmed - (start_mh.Cured + start_mh.Deaths))
        R0 = (start_mh.Cured)
        D0 = start_mh.Deaths
        simulator.integrate(
            timesteps=data.shape[0]-1,
            t_min=1,
            start_date=start_date.date(),
            S=population-I0,
            E=0,
            Is=int(I0*config["p_a"]),
            Ia=int(I0*(1 - config["p_a"])),
            R=R0,
            D=D0
        )

        projected_results = pd.DataFrame({
            "Confirmed": (simulator.Ia + simulator.Is + start_mh.Confirmed),
            "Cured": simulator.R,
            "Deaths": simulator.D
        })

        y_pred = projected_results.to_numpy() / population
        y_actual = data.loc[data.Date > start_date, ["Confirmed", "Cured", "Deaths"]].to_numpy() / population

        tune.report(kl_div=np.mean(np.sum(y_pred*np.log(y_pred/y_actual), axis=-1)))

    analysis = tune.run(
        _tuning_fn,
        config={
            "beta": tune.uniform(0.75, 1.2),
            "p_a": tune.uniform(0.25, 0.7),
            "r_b": tune.uniform(0.3, 0.8),
            "eps": tune.uniform(1. / 15., 1. / 3.),
            "mu": tune.uniform(0.01, 0.05),
            "rho": tune.loguniform(1e-4, 1e-3),
            "p_d": tune.uniform(0.01, 0.05),
        },
        metric="kl_div",
        mode="min",
        num_samples=num_samples,
        # search_alg=ConcurrencyLimiter(
        #     DragonflySearch(
        #         optimizer="bandit",
        #         domain="euclidean"
        #     ),
        #     max_concurrent=12
        # ),
        # scheduler=AsyncHyperBandScheduler()
    )

    print(
        "Best config: ", 
        analysis.get_best_config(metric="kl_div", mode="min")
    )
    return analysis
