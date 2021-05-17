"""Epidemic model.

@author Bruno Goncalves, Mandar Chandorkar
"""

from datetime import datetime, timedelta
import networkx as nx
import numpy as np
from numpy import linalg
from numpy import random
import scipy.integrate
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
tqdm.pandas()


class EpiModel(object):
    """Simple Epidemic Model Implementation.

    Provides a way to implement and numerically integrate
    an epidemic model.
    """

    def __init__(self, compartments=None):
        """Create a new model."""
        self.transitions = nx.MultiDiGraph()
        self.locality = None
        self.locality_period = None

        if compartments is not None:
            self.transitions.add_nodes_from([comp for comp in compartments])

    def add_interaction(self, source, target, agent, rate):
        """Add an interaction."""
        self.transitions.add_edge(source, target, agent=agent, rate=rate)

    def add_spontaneous(self, source, target, rate):
        """Add a spontaneous conversion."""
        self.transitions.add_edge(source, target, rate=rate)

    def _new_cases(self, population, time, pos):
        """Compute new cases for the integration routine."""
        diff = np.zeros(len(pos))
        N = np.sum(population)

        for edge in self.transitions.edges(data=True):
            source = edge[0]
            target = edge[1]
            trans = edge[2]

            rate = trans['rate']*population[pos[source]]

            if 'agent' in trans:
                agent = trans['agent']
                rate *= population[pos[agent]]/N

                if self.locality is not None:
                    curr_t = int(time) // self.locality_period
                    if curr_t < len(self.locality):
                        season = float(self.locality[curr_t])
                    else:
                        season = float(mean(self.locality))
                    rate *= season

            diff[pos[source]] -= rate
            diff[pos[target]] += rate

        return diff

    def plot(self, title=None, normed=True, **kwargs):
        """Plot model results."""
        try:
            if normed:
                N = self.values_.iloc[0].sum()
                ax = (self.values_/N).plot(**kwargs)
            else:
                ax = self.values_.plot(**kwargs)

            ax.set_xlabel('Time')
            ax.set_ylabel('Population')

            if title is not None:
                ax.set_title(title)
            ax.minorticks_on()
            ax.grid(True, which='major', linestyle='-', linewidth=0.25, axis='both', alpha=0.85, color="black")
            ax.grid(True, which='minor', linestyle='-.', linewidth=0.25, axis='both', alpha=0.7)
            return ax
        except Exception:
            raise Exception('You must call integrate() first')

    def __getattr__(self, name):
        """Return the individual compartment values."""
        if 'values_' in self.__dict__:
            return self.values_[name]
        else:
            raise AttributeError("'EpiModel' object has no attribute '%s'" % name)

    def simulate(
            self,
            timesteps,
            t_min=1,
            locality=None,
            locality_period: int = 7,
            start_date=datetime.today().date(),
            **kwargs):
        """Simulate the epidemic model."""
        pos = {comp: i for i, comp in enumerate(kwargs)}
        population = np.zeros(len(pos), dtype='int')

        for comp in pos:
            population[pos[comp]] = kwargs[comp]

        values = []
        values.append(population)

        comps = list(self.transitions.nodes)
        time = np.arange(t_min, t_min+timesteps, 1, dtype='int')
        start_date = (
            start_date +
            timedelta(days=t_min)
        )
        dates = pd.date_range(start=start_date, end=start_date+timedelta(days=timesteps-1), freq='D')

        self.locality = locality
        self.locality_period = locality_period

        for t in time:
            pop = values[-1]
            new_pop = values[-1].copy()
            N = np.sum(pop)

            for comp in comps:
                trans = list(self.transitions.edges(comp, data=True))

                prob = np.zeros(len(comps), dtype='float')

                for _, node_j, data in trans:
                    source = pos[comp]
                    target = pos[node_j]

                    rate = data['rate']

                    if 'agent' in data:
                        agent = pos[data['agent']]
                        rate *= pop[agent]/N

                        if self.locality is not None:
                            curr_t = int(t) // self.locality_period
                            if curr_t < len(self.locality):
                                season = float(self.locality[curr_t])
                            else:
                                season = float(mean(self.locality))
                            rate *= season

                    prob[target] = rate

                prob[source] = 1-np.sum(prob)

                delta = random.multinomial(pop[source], prob)
                delta[source] = 0

                changes = np.sum(delta)

                if changes == 0:
                    continue

                new_pop[source] -= changes

                for i in range(len(delta)):
                    new_pop[i] += delta[i]

            values.append(new_pop)

        values = np.array(values)
        self.values_ = pd.DataFrame(values[1:], columns=comps, index=dates)

    def integrate(
            self,
            timesteps,
            t_min=1,
            locality=None,
            locality_period: int = 7,
            start_date=datetime.today().date(),
            **kwargs):
        """Integrate the epidemic model."""
        pos = {comp: i for i, comp in enumerate(kwargs)}
        population = np.zeros(len(pos))

        for comp in pos:
            population[pos[comp]] = kwargs[comp]

        time = np.arange(t_min, t_min+timesteps, 1)
        start_date = (
            start_date +
            timedelta(days=t_min)
        )
        dates = pd.date_range(start=start_date, end=start_date+timedelta(days=timesteps-1), freq='D')

        self.locality = locality
        self.locality_period = locality_period
        self.values_ = pd.DataFrame(
            scipy.integrate.odeint(self._new_cases, population, time, args=(pos,)),
            columns=pos.keys(),
            index=dates
        )

    def __repr__(self):
        """Print out the model."""
        text = 'Epidemic Model with %u compartments and %u transitions:\n\n' % \
            (
                self.transitions.number_of_nodes(),
                self.transitions.number_of_edges()
            )

        for edge in self.transitions.edges(data=True):
            source = edge[0]
            target = edge[1]
            trans = edge[2]

            rate = trans['rate']

            if 'agent' in trans:
                agent = trans['agent']
                text += "%s + %s = %s %f\n" % (source, agent, target, rate)
            else:
                text += "%s -> %s %f\n" % (source, target, rate)

        R0 = self.R0()

        if R0 is not None:
            text += "\nR0=%1.2f" % R0

        return text

    def _get_active(self):
        active = set()

        for node_i, node_j, data in self.transitions.edges(data=True):
            if "agent" in data:
                active.add(data['agent'])
            else:
                active.add(node_i)

        return active

    def _get_susceptible(self):
        susceptible = set()

        for node_i, node_j, data in self.transitions.edges(data=True):
            if "agent" in data:
                susceptible.add(node_i)

        return susceptible

    def _get_infections(self):
        inf = {}

        for node_i, node_j, data in self.transitions.edges(data=True):
            if "agent" in data:
                agent = data['agent']

                if agent not in inf:
                    inf[agent] = {}

                if node_i not in inf[agent]:
                    inf[agent][node_i] = {}

                inf[agent][node_i]['target'] = node_j
                inf[agent][node_i]['rate'] = data['rate']

        return inf

    def R0(self):
        """Compute R0."""
        infected = set()

        susceptible = self._get_susceptible()

        for node_i, node_j, data in self.transitions.edges(data=True):
            if "agent" in data:
                infected.add(data['agent'])
                infected.add(node_j)

        infected = sorted(infected)
        N_infected = len(infected)

        F = np.zeros((N_infected, N_infected), dtype='float')
        V = np.zeros((N_infected, N_infected), dtype='float')

        pos = dict(zip(infected, np.arange(N_infected)))

        try:
            for node_i, node_j, data in self.transitions.edges(data=True):
                rate = data['rate']

                if "agent" in data:
                    target = pos[node_j]
                    agent = pos[data['agent']]

                    if node_i in susceptible:
                        F[target, agent] = rate
                else:
                    source = pos[node_i]

                    V[source, source] += rate

                    if node_j in pos:
                        target = pos[node_j]
                        V[target, source] -= rate

            eig, v = linalg.eig(np.dot(F, linalg.inv(V)))

            return eig.max()
        except Exception:
            return None


if __name__ == '__main__':

    beta = 0.2
    mu = 0.1

    SIR = EpiModel()
    SIR.add_interaction('S', 'I', 'I', beta)
    SIR.add_spontaneous('I', 'R', mu)

    N = 100000
    I0 = 10

    season = np.ones(365+1)
    season[74:100] = 0.25

    fig, ax = plt.subplots(1)

    Nruns = 1000
    values = []

    for i in tqdm(range(Nruns), total=Nruns):
        SIR.simulate(365, season, S=N-1, I=1, R=0)

        ax.plot(SIR.I/N, lw=.1, c='b')

        if SIR.I.max() > 10:
            values.append(SIR.I)

    values = pd.DataFrame(values)
    (values.median(axis=0)/N).plot(ax=ax, c='r')

    fig.savefig('SIR.png')
