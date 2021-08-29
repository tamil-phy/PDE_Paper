import os
import hashlib
import json

logging_format = "%(levelname)-8s:%(filename)s.%(funcName)20s >>   %(message)s",
config = None

def hash_config(config):
    sha256_hash = hashlib.sha256()
    temppath = '/tmp/temp_temp.json'
    json.dump(config,
              open(temppath, 'w'),
              indent=4,
              ensure_ascii=False)
    
    with open(temppath, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096),b""):
            sha256_hash.update(byte_block)
            
    return sha256_hash.hexdigest()

def init_config(aconfig, hpconfig):
    global config
    config = aconfig
    config['hash'] = config['hpconfig_name'] + '__' + hash_config(hpconfig)[-6:]
    os.makedirs(config['hash'], exist_ok=True)
    for k, v in config['metrics_path'].items():
        config['metrics_path'][k] = '{}/{}'.format(config['hash'], v)

def get_directory_from_file(path):
    return os.path.basename(
        os.path.dirname(
            os.path.realpath(path)))

def get_dataset_path_from_file(path):
    print('generating dataset path for {}'.format(path))
    output_path = '{}/{}.pkl'.format(config['DATA_DIR'],
                                     get_directory_from_file(path))
    return output_path

def get_weights_path_from_file(path):
    print('generating weights path for {}'.format(path))
    output_path = '{}/{}.pt'.format(config['hash'],
                                       get_directory_from_file(path))
    return output_path

