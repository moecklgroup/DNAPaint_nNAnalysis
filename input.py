# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:47:05 2025

@author: sfritsc
"""

import json

variables = {}
with open(r"C:\Users\sfritsc\Desktop\inputfile.json") as f:
    variables = json.load(f)

# get the values from the dictionary
path = variables.get('path')
number = variables.get('number')
secondnumber = variables.get('second_number')

print(path)
print(number*secondnumber)