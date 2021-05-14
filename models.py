"""Model definitions.

@author Mandar Chandorkar
"""

from epimodel import *


def seird(
        beta: float,
        eps: float,
        mu: float,
        rho: float,
        p_d: float) -> EpiModel:
    """Create an SEIRD model.

    An SEIRD model has the following components.
     - `S`: Suceptible.
     - `E`: Exposed, after coming in contact with an infectious host.
     - `I`: Infected, symptomatic.
     - `R`: Recovered.
     - `D`: Dead.

    Args:
        beta: The exposure rate, when in contact with infected hosts.
        eps: The incubation frequency, (1 / incubation period).
        mu: The recovery rate.
        rho: The re-infection rate.
        p_d: The death rate.

    Returns: An `EpiModel` instance.
    """
    model_instance = EpiModel(
        compartments=['S', 'E', 'I', 'R', 'D']
    )
    model_instance.add_interaction('S', 'E', 'I', beta)

    model_instance.add_spontaneous('E', 'I', eps)

    model_instance.add_spontaneous('I', 'R', mu*(1-p_d))
    model_instance.add_spontaneous('R', 'S', rho)
    model_instance.add_spontaneous('I', 'D', mu*p_d)
    return model_instance


def seiird(
        beta: float,
        p_a: float,
        r_b: float,
        eps: float,
        mu: float,
        rho: float,
        p_d: float) -> EpiModel:
    """Create an SEIIRD model.

    An SEIIRD model has the following components.
     - `S`: Suceptible.
     - `E`: Exposed, after coming in contact with an infectious host.
     - `Is`: Infected, symptomatic.
     - `Ia`: Infected, asymptomatic.
     - `R`: Recovered.
     - `D`: Dead.

    Args:
        beta: The exposure rate, when in contact with symptomatic hosts.
        p_a: Fraction of infected hosts which are asymptomatic.
        r_b: Fraction of the exposure rate `beta`, when in contact with asymptomatic hosts.
        eps: The incubation frequency, (1 / incubation period).
        mu: The recovery rate.
        rho: The re-infection rate.
        p_d: The death rate for symptomatic cases.

    Returns: An `EpiModel` instance.
    """
    model_instance = EpiModel(
        compartments=['S', 'E', 'Is', 'Ia', 'R', 'D']
    )
    model_instance.add_interaction('S', 'E', 'Is', beta)
    model_instance.add_interaction('S', 'E', 'Ia', beta*r_b)

    model_instance.add_spontaneous('E', 'Is', eps*(1-p_a))
    model_instance.add_spontaneous('E', 'Ia', eps*p_a)

    model_instance.add_spontaneous('Ia', 'R', mu)
    model_instance.add_spontaneous('Is', 'R', mu*(1-p_d))
    model_instance.add_spontaneous('R', 'S', rho)
    model_instance.add_spontaneous('Is', 'D', mu*p_d)
    return model_instance
