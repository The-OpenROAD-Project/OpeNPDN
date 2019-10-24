#BSD 3-Clause License
#
#Copyright (c) 2019, The Regents of the University of Minnesota
#
#All rights reserved.
#
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
from construct_eqn import construct_eqn
import numpy as np
from T6_PSI_settings import T6_PSI_settings
"""
Created on Thu Mar 21 21:16:37 2019

@author: Vidya A Chhabria
"""

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
    
