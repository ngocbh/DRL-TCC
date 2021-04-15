# odmc-wrsn
On-demand Mobile Charger in Wireless Rechargeable Sensor Network

## Usage

```
python main.py --help
```

```sh
usage: main.py [-h] [--num_sensors NUM_SENSORS] [--num_targets NUM_TARGETS] [--mode {train,eval}] [--config CONFIG] [--checkpoint CHECKPOINT] [--save_dir SAVE_DIR]
               [--epoch_start EPOCH_START] [--render] [--verbose]

Mobile Charger Trainer

optional arguments:
  -h, --help            show this help message and exit
  --num_sensors NUM_SENSORS, -ns NUM_SENSORS
  --num_targets NUM_TARGETS, -nt NUM_TARGETS
  --mode {train,eval}
  --config CONFIG, -cf CONFIG
  --checkpoint CHECKPOINT, -cp CHECKPOINT
  --save_dir SAVE_DIR, -sd SAVE_DIR
  --epoch_start EPOCH_START
  --render, -r
  --verbose, -v
```
## Simulation
To run simulation:

```sh
python main.py --mode eval --render --verbose -cp checkpoint_path -cf config_path
```
For examples:

```sh
python main.py --checkpoint checkpoints/mc_20_10_0/21 --config configs/mc_20_10_0.yml --mode eval --render --verbose
```

## Acknowledgements

Parts of the code are based on the following source-codes:

* https://github.com/mveres01/pytorch-drl4vrp
* https://github.com/openai/gym
* https://github.com/ikostrikov/pytorch-a3c
* https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail 
* https://github.com/DLR-RM/stable-baselines3
* https://github.com/ahottung/NLNS.
