import pytest
import numpy as np
from T6_PSI_settings import T6_PSI_settings
import json

def test_load_json():
    settings_obj = T6_PSI_settings()
    test_data = settings_obj.load_json('test/test.json')
    exp_data ={}
    exp_data['test'] ={}
    exp_data['test']['1'] ={}
    exp_data['test2'] ={}
    exp_data['test2']['2'] ={}
    exp_data['test']['1'] = 5
    exp_data['test2']['2'] = 'test'
    assert exp_data == test_data


#TODO chcek that the json has alll the data is it automatically checked given
#that calling setings obj calls it.  
#TODO check if the content s of the json seems resonable? 


