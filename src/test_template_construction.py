import pytest
from template_construction import template_def
import numpy as np

@pytest.fixture
def template_2():
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
def template():
    template = template_def(
        2,
        np.array([1e-6,2e-6]),
        np.array([1e-6,2e-6]),
        np.array([1e-9,1e-9]),
        np.array([1e-9,1e-9]),
        np.array([0,1]),
        10e-6,
        10e-6
        )
    return template 



def test_get_y_grid_pitch_gcd(template):
    pitches = np.array([25e-9,45e-9,35e-9,65e-9])
    dirs = np.array([0,1,0,1])
    val = template.get_y_grid_pitch(pitches,dirs)
    assert (val - 6e-9) < 1e-12

def test_get_x_grid_pitch_gcd(template):
    pitches = np.array([25e-9,48e-9,35e-9,66e-9])
    dirs = np.array([0,1,0,1])
    val = template.get_x_grid_pitch(pitches,dirs)
    assert (val - 6e-9) < 1e-12

def test_numx(template_2):
    assert template_2.num_x == 40

def test_numy(template_2):
    assert template_2.num_y == 79

def test_create_init_offset(template_2):
    offset = template_2.create_init_offset()
    expected = np.array([0,0,1,2])
    for o,off in enumerate(offset):
        assert off == expected[o], (
            "expected value for layer %d is %4.3e got %4.3e"%(
            o,expected[o],off))
        
def test_check_index(template_2):
    res = template_2.check_index(0,0,-1,0,1,1,1)
    assert res==0
    res = template_2.check_index(0,0,0,-1,1,1,1)
    assert res==0
    res = template_2.check_index(0,0,40,0,1,1,1)
    assert res==0
    res = template_2.check_index(0,0,0,79,1,1,1)
    assert res==0
    #TODO check offset matchin for connection.

def test_convert_index(template_2):
    res = template_2.convert_index(-1,0)
    assert res==0
    res = template_2.convert_index(0,-1)
    assert res==0
    res = template_2.convert_index(40,0)
    assert res==0
    res = template_2.convert_index(0,79)
    assert res==0
    res = template_2.convert_index(10,5)
    assert res==210

def test_conductivity(template_2):
    res = template_2.conductivity(-1,0,0,0,0)
    assert res==0
    res = template_2.conductivity(0,-1,0,0,0)
    assert res==0
    res = template_2.conductivity(40,0,0,0,0)
    assert res==0
    res = template_2.conductivity(0,79,0,0,0)
    assert res==0
    res = template_2.conductivity(10,5,100e-9,10e-3,128e-9)
    assert (res-781.25)<1e-9

    #TODO check G matrix single ayer G and connect via

def test_create_G_start(template_2):
    expected = np.array([0,3160,6320,9480])
    expected = expected.reshape((4,1))
    start = template_2.create_G_start()
    assert np.array_equal(start,expected)
