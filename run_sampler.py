"""Parameter sampling and tuning.

@author Mandar Chandorkar
"""

from typing import Any, Dict
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
from ray import tune
import models
import data_utils


def load_data(state: str, start_date: datetime.datetime) -> pd.DataFrame:
    """Load data for a state."""
    df = data_utils.load_covid_data_india().get(state)
    return df.loc[df.Date >= start_date].copy()


def tuning_exp(
        state: str,
        population: int,
        start_date: datetime.datetime,
        hyp_config: Dict[str, Any],
        num_samples: int = 50):
    """Perform the sampling experiment."""
    def _tuning_fn(config: Dict[str, Any]):
        data: pd.DataFrame = load_data(state, start_date)
        simulator = models.seird(
            beta=config["beta"],
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
            I=I0,
            R=R0,
            D=D0
        )

        projected_results = pd.DataFrame({
            "rest": population - (simulator.I + simulator.D),
            "infected": simulator.I,
            "dead": simulator.D
        })

        y_pred = projected_results.to_numpy() / population
        # y_pred_aug = np.concatenate(
        #     [
        #         1.0 - np.sum(y_pred, axis=-1, keepdims=True),
        #         y_pred
        #     ],
        #     axis=-1
        # )

        case_data: pd.DataFrame = data.loc[
            data.Date > start_date,
            ["Confirmed", "Cured", "Deaths"]
        ]
        projected_case_data = pd.DataFrame(
            {
                "rest": population - (case_data["Confirmed"] - case_data["Cured"] + case_data["Deaths"]),
                "infected": case_data["Confirmed"] - case_data["Cured"],
                "dead": case_data["Deaths"]
            }
        )

        y_actual = projected_case_data.to_numpy() / population
        # y_actual_aug = np.concatenate(
        #     [
        #         1.0 - np.sum(y_actual, axis=-1, keepdims=True),
        #         y_actual
        #     ],
        #     axis=-1
        # )

        tune.report(
            kl_div=np.mean(np.sum(y_pred*np.log(y_pred/y_actual), axis=-1))
        )

    analysis = tune.run(
        _tuning_fn,
        config=hyp_config,
        metric="kl_div",
        mode="min",
        num_samples=num_samples,
        local_dir=Path.cwd() / ".tuning",
        verbose=2
    )

    print(
        "Best config: ",
        analysis.get_best_config(metric="kl_div", mode="min")
    )
    return analysis
