**Status:** Archive (code is provided as-is, no updates expected)

# Meta-Learning Shared Hierarchies

Code for [Meta-Learning Shared Hierarchies](https://s3-us-west-2.amazonaws.com/openai-assets/MLSH/mlsh_paper.pdf).


##### Installation

```
Add to your .bash_profile (replace ... with path to directory):
export PYTHONPATH=$PYTHONPATH:/.../mlsh/gym;
export PYTHONPATH=$PYTHONPATH:/.../mlsh/rl-algs;

Install MovementBandits environments:
cd test_envs
pip install -e .
```

##### Running Experiments
```
python main.py --task AntBandits-v1 --num_subs 2 --macro_duration 1000 --num_rollouts 2000 --warmup_time 20 --train_time 30 --replay False AntAgent

```
Once you've trained your agent, view it by running:
```
python main.py [...] --replay True --continue_iter [your iteration] AntAgent
```
The MLSH script works on any Gym environment that implements the randomizeCorrect() function. See the envs/ folder for examples of such environments.

To run on multiple cores:
```
mpirun -np 12 python main.py ...
```
