# Sampling based MPC 

Run ```python setup.py develop``` to setup the repo

Run scenarios from the ```examples``` dir

The main source code is in the ```behavior_kit``` dir

## Running the MPC in different environments
Inside `examples/ros2/`, build the ros workspace using `colcon build`. Source the workspace in each new terminal, source the workspace using `. install/setup.bash`.
To change th number of goals, change the variabele `num_goals` in both the CPP optimizer and Python plotting code.
### NGSIM
#### SMPC
1. Source the workspace and run `ros2 run acado_ros combined_ngsim`.
2. In a new terminal, source the workspace and run `ros2 run simulation_env ngsim_env`.
#### Multi-Goal Acado
1. Change the number goals in `src/acado_ros/src/combined_ngsim.cpp` and `src/simulation_env/simulation_env/ngsim_acado.py` by changing the variable `num_goals` to a value geater than 1.
2. Source the workspace and run `ros2 run acado_ros combined_ngsim`.
3. In a new terminal, source the workspace and run `ros2 run simulation_env ngsim_acado`.
#### Single-Goal Acado
1. Change the number goals in `src/acado_ros/src/combined_ngsim.cpp` and `src/simulation_env/simulation_env/ngsim_acado.py` by changing the variable `num_goals` to 1.
2. Source the workspace and run `ros2 run acado_ros combined_ngsim`.
3. In a new terminal, source the workspace and run `ros2 run simulation_env ngsim_acado`.

### Combined Environment
#### SMPC
1. Source the workspace and run `ros2 run acado_ros combined_mpc_multi`.
2. In a new terminal, source the workspace and run `ros2 run simulation_env combined_sim_new`.
#### Multi-Goal Acado
1. Change the number goals in `src/acado_ros/src/combined_mpc_multi.cpp` and `src/simulation_env/simulation_env/combined_sim_acado.py` by changing the variable `num_goals` to a value geater than 1.
2. Source the workspace and run `ros2 run acado_ros combined_ngsim`.
3. In a new terminal, source the workspace and run `ros2 run simulation_env combined_sim_acado`.
#### Single-Goal Acado
1. Change the number goals in `src/acado_ros/src/combined_mpc_multi.cpp` and `src/simulation_env/simulation_env/combined_sim_acado.py` by changing the variable `num_goals` to 1.
2. Source the workspace and run `ros2 run acado_ros combined_ngsim`.
3. In a new terminal, source the workspace and run `ros2 run simulation_env combined_sim_acado`. ngsim_acado`.