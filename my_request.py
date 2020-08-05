# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:35:31 2020

@author: Krishna
"""

import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'name':'E_500_Avantgarde_AMG_Ausstattung', 'vehicleType':'limousine', 'yearOfRegistration':2002,'gearbox':'automatik','powerPS':306,'model':'e_klasse','kilometer':150000,'fuelType':'nan','brand':'mercedes_benz','notRepairedDamage':'nan'})
print(r.json())

