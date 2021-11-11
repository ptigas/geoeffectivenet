# Running the codes.

1. First of all ensure the data is downloaded by running the bash script `get_data.sh` in 'scripts/`. 
2. Move the data to the main repo as `mv data_local ../`
3. Install all dependencies as `pip install -r requirements.txt`.
4. Then, for running the parameter sweep, do: `wandb sweep sweep.yaml`, which will give you a sweep_id (actually a command as wandb agent ...)
5. Run the command `wandb agent <sweep_id>`

Reference for the sweeps: https://github.com/borisdayma/lightning-kitti. 

The data and model checkpoints are in "geo2020_supermag/Uniformdata_multiyear/" in the bucket.