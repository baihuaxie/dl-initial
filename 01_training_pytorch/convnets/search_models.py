""" Run multiple models """

import argparse
import os

import utils
import launch


parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='./experiments/07022020',
                    help='Directory containing params.json')
parser.add_argument('--data_dir', default='./data/', help="Directory containing the dataset")


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Perform hypersearch over one parameter
    models = [
        'densenet40_k12',
        'densenet100_k12',
        'densenet100_k24',
        'densenetbc100_k12',
        'densenetbc250_k24',
        'densenetbc190_k40'
    ]

    for model in models:
        # Modify the relevant parameter in params
        params.model = model

        # Launch job (name has to be unique)
        job_name = "{}".format(model)
        launch.launch_training_job(args.parent_dir, args.data_dir, job_name, params)
