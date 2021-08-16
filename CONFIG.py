import os


DATA_DIR = '/home/vanangamudi/agam/projects/code/saama/PDE_Paper/data'
WEIGHTS_DIR = '/home/vanangamudi/agam/projects/code/saama/PDE_Paper/weights'


def get_directory_from_file(path):
    return os.path.basename(
        os.path.dirname(
            os.path.realpath(path)))

def get_dataset_path_from_file(path):
    print('generating dataset path for {}'.format(path))
    output_path = '{}/{}.pkl'.format(DATA_DIR, get_directory_from_file(path))
    return output_path

def get_weights_path_from_file(path):
    print('generating weights path for {}'.format(path))
    output_path = '{}/{}.pt'.format(WEIGHTS_DIR, get_directory_from_file(path))
    return output_path
