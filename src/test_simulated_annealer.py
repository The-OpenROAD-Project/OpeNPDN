import pytest
from template_construction import template_def
from simulated_annealer import simulated_annealer
import numpy as np

@pytest.fixture
def template():
    template = template_def(
        4,
        np.array([128e-9,512e-9,512e-9,1792e-9]),
        np.array([1e-9,2e-9, 3e-9,4e-9]),
        np.array([1e-9,1e-9,1e-9,2e-9]),
        np.array([1e-9,1e-9,1e-9,2e-9]),
        np.array([0,1,0,1]),
        10e-6,
        10e-6
        )
    return template 

@pytest.fixture
def annealer():
    annealer = simulated_annealer([0,0,0,0],
        100,
        0.1,
        0.9,
        5,
        0)
    return annealer
#TODO sim anneal, energy, delta energy 
def test_cool_down(annealer):
    res = annealer.cool_down(5,0.7)
    assert np.isclose(res,3.5)

def test_acceptMove(annealer):
    res = annealer.acceptMove(-0.000001,10000000)
    assert res == 1

def test_move(annealer):
    state = [0,1,2,1]
    temps = [0,1,2,3]
    res = annealer.move(state,temps)
    assert res != state

#needs a proper setup
#def test_pdn_util(template,annealer):
#    template_list = [template]
#    temp_util = annealer.pdn_util(template_lists,0,)
    


