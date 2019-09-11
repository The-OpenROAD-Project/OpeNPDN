# OpeNPDN: Neural networks for automated synthesis of Power Delivery Networks (PDN)

## Instructions for populating the tool_config.json file. 

This file contains the parameters that control certain features of the tool. Examples include, controlling the number of parallel processes run during simulated annealing, maximum current value for a given PDK, distribution of C4 bumps etc.


The following are a list of variables defined in this file:

- num_vdd_per_region: number of C4 bumps in a given region. 
- current_map_num_regions: this variable controls the number of regions in the x
  and y-direction each that are extracted from the training set current maps
- num_maps: number of current and congestion maps used in the training set
- start_maps: if data generation is interrupted midway, it is possible to restart the training data generation from this map ID.
- num_parallel_runs: number of processes to run in parallel for training data generation. Ensure this number is less that the value returned by *nproc* command.
- num_per_run: this sets the number of maps to run for every parallel process. Recommended value is 1. 
- validation_percent: percentage of data generated that is separated out as the validation set.
- test_percent: percentage of the data generated that is separated out as the test set
- current_offset: current map amplitude offset. This parameter is tuned while generating the new training set for a new technology.
- current_scaling: factor that tunes the maximum current value for a new technology.
- N_EPOCHS: number of epochs for training the CNN.
- max_current: maximum current value for a given PDK.

* For most parts of a regular run of the flow these parameters need not be changed, but are still made available to a user to use at their discretion.
