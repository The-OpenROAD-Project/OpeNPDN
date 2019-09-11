#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun June 2 15:10:01 2019
This script runs the inference flow using the trained optimized CNN to predict
the class for a given testcase
@author: chhab011
"""

import json

with open("input/template_definition.json", 'r') as JSON:
       dict = json.load(JSON)


for layer in dict['property']['TECH_layers']:
        attributes = dict['layers'][layer]
        dict['layers'][layer]['width'] = attributes['width']/4
        dict['layers'][layer]['min_width'] = attributes['min_width']/4
        for i,pitch in enumerate(attributes['pitch']):
            dict['layers'][layer]['pitch'][i] = pitch/4
        dict['layers'][layer]['via_res'] = attributes['via_res']/4
        dict['layers'][layer]['res'] = attributes['res']/4
        dict['layers'][layer]['t_spacing'] = attributes['t_spacing']/4

with open("template_definition_scaled.json", 'w') as outfile:
    json.dump(dict, outfile, indent=4)

