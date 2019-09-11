import pytest
from template_construction import template_def
from construct_eqn import construct_eqn
import numpy as np
from T6_PSI_settings import T6_PSI_settings

#@pytest.fixture
#def template_2():
#    template = template_def(
#        4,
#        np.array([128e-9,512e-9,512e-9,1792e-9]),
#        np.array([1e-9,2e-9, 3e-9,4e-9]),
#        np.array([1e-9,1e-9,1e-9,2e-9]),
#        np.array([1e-9,1e-9,1e-9,2e-9]),
#        np.array([0,1,0,1]),
#        10e-6,
#        10e-6
#        )
#    return template 
#
#@pytest.fixture
#def template():
#    template = template_def(
#        2,
#        np.array([1e-6,2e-6]),
#        np.array([1e-6,2e-6]),
#        np.array([1e-9,1e-9]),
#        np.array([1e-9,1e-9]),
#        np.array([0,1]),
#        10e-6,
#        10e-6
#        )
#    return template 

#TODO how to override json
#def test_get_regional_current():
#    eq = construct_eqn();
#    reg_cur , cur_row = get_regional_current(self, chip_current, ind_x, ind_y):

def test_connect_resistors():
    eq = construct_eqn()
    G = np.zeros((3,3))+np.eye(3)
    G_new =  eq.connect_resistors( G, 1, 2, 5)
    assert ~(np.array_equal(G_new,G))
    G_exp  = np.array([[1,0,0],[0,6,-5],[0,-5,6]]) 
    assert np.array_equal(G_exp,G)
    
