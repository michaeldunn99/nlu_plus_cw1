#!/bin/bash

# Define possible values for each hyperparameter
param1_mode_values=("rnn_vs_gru_np")
param2_directory_values=("data")
param3_hiddendims_values=(50)
param4_lookback_values=(0 2 5)
param5_learning_values=(0.5)
param6_epochs_values=(10)
param7_annealing_values=(5)

counter=0
# Iterate over all combinations of hyperparameter values
for param1 in "${param1_mode_values[@]}"; do
    for param2 in "${param2_directory_values[@]}"; do
        for param3 in "${param3_hiddendims_values[@]}"; do
            for param4 in "${param4_lookback_values[@]}"; do
                for param5 in "${param5_learning_values[@]}"; do
                    for param6 in "${param6_epochs_values[@]}"; do
                        for param7 in "${param7_annealing_values[@]}"; do
                        # Determine the appropriate redirection operator
                            if [ $counter -eq 0 ]; then
                                # First iteration: overwrite the file
                                python code/runner.py "$param1" "$param2" "$param3" "$param4" "$param5" "$param6" "$param7" > output/hyper_param_exploration.txt
                            else
                                # Subsequent iterations: append to the file
                                python code/runner.py "$param1" "$param2" "$param3" "$param4" "$param5" "$param6" "$param7" >> output/hyper_param_exploration.txt
                            fi
                            # Increment the counter
                            ((counter++))
                        done
                    done
                done
            done
        done
    done
done