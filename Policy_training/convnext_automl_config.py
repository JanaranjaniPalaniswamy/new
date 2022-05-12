import argparse
import json
import os

from plai.automl.single_objective import run


_description = """
Run AutoML single-objective optimization with BOHB.
Please specify all parameters directly in this scipt.
"""

worker_config = {
    "hyperparameters": [
        "learning_rate",
        "weight_decay",
        "scheduler_relative_final_lr",
        # "clip_grad_norm_to",
    ],
    "policy_config": {
        "path_to_class": "plai.learning.policies.FlexibleCnnBackboneWaypointsDetectionHead",
        "path_to_state_dict": None,
        "config": {
            "input_names": [
                "binary_grid_features"
            ],
            "input_shapes": "INFER",
            "output_names": [
                "waypoints",
                "predictions_3s"
            ],
            "output_shapes": "INFER",
            "random_seed": None,
            "num_detection_frames": 1,
            "model": "convnext_base",
            "truncate_after": "features.1"
        }
    },
    "training_config": "DEFAULT",
    "optimization_target": "dev loss EMA",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=_description)
    parser.parse_args()

    parameters = {
        "min_budget": 1 / 2.5 ** 2 - 1e-3,
        "max_budget": 1,
        "eta": 2.5,
        "num_iterations": 10,  # 6@eta=2.4 -> 15 full runs, 10@eta=3 -> 27 full runs, 20@eta=2.5 -> 50 full runs
        "output_path": "/mnt/workspace/experiments_jana/convNext/2022-04-29_extendedVis/automl",
        "dataset": None,  # or use worker_config["training_config"]["dataset_paths"] if training config is provided,
        "name": "convnext_base",
        "worker_config": worker_config,
    }

    run(**parameters)

    with open(os.path.join(parameters["output_path"], "parameters.json"), "w") as fh:
        json.dump(parameters, fh, indent=2)
