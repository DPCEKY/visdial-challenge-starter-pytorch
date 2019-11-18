#!/usr/bin/env python
# coding: utf-8


import json

def sample_data():

	## Read and visualize data
	train_f = 'data/visdial_1.0_train.json'

	with open(train_f) as f:
	    data = json.load(f)

	data.keys()

	len(data['data']['questions'])
	data['data']['questions'][:5]

	len(data['data']['answers'])
	data['data']['answers'][:5]

	len(data['data']['dialogs'])
	data['data']['dialogs'][0]


	## Sample data
	sample_rate = 0.1

	num_dialogs = len(data['data']['dialogs'])
	
	data['data']['dialogs'] = data['data']['dialogs'][0:int(num_dialogs * sample_rate)]

	train_f = 'data/visdial_1.0_train_sampled.json'

	with open(train_f, 'w') as outfile:
	    json.dump(data, outfile)


if __name__ == "__main__":
	sample_data()