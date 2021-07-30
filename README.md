# Running the codes.

First of all ensure the data is downloaded by running the bash script `get_data.sh` in 'scripts/`. 

Then, for running the parameter sweep, do:

`wandb sweep sweep.yaml`, which will give you a sweep_id. 

`wandb agent <sweep_id>`

Reference for the sweeps: https://github.com/borisdayma/lightning-kitti. 