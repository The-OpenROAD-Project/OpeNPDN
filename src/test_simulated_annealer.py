#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#* Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
#* Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#* Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import pytest
from template_construction import template_def
fro"""
Created on Thu Mar 21 21:16:37 2019

@author: Vidya A Chhabria
"""
m construct_eqn import construct_eqn
import numpy as np
from T6_PSI_settings import T6_PSI_settings


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
    


