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
"""
Created on Thu Mar 21 21:16:37 2019

@author: Vidya A Chhabria
"""

import pytest
from template_construction import template_def
from construct_eqn import construct_eqn
import numpy as np
from T6_PSI_settings import T6_PSI_settings


import pytest
import numpy as np
import current_map_generator
#@pytest.fixture

def test_read_power_report():
    pwr_rpt = current_map_generator.read_power_report("test/test_pwr_rpt.txt")
    assert 'fifo' in pwr_rpt , "Instance fifo not found in test report"
    if 'fifo' in pwr_rpt :
        assert pwr_rpt['fifo']['cell'] == "NAND2"
        assert pwr_rpt['fifo']['lib']  == "lib_file.lib"
        assert np.isclose(pwr_rpt['fifo']['internal_power'] ,2.765e-8) 
        assert np.isclose(pwr_rpt['fifo']['switching_power'],3.674e-8)
        assert np.isclose(pwr_rpt['fifo']['leakage_power']  ,0.000e-8)
        assert np.isclose(pwr_rpt['fifo']['total_power']    ,6.440e-8)

