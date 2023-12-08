import os
from os.path import join, dirname, realpath, exists, abspath, expandvars
import json
import numpy as np
import inspect


def to_list(obj):
    obj = [obj] if type(obj) != list else obj
    return obj


def is_interactive():
    try:
        __IPYTHON__
    except NameError:
        return False
    else:
        return True


class SearchAttribute():

    def __init__(self, search_name):
        self.search_name = search_name
        self.found = False
        self.searched = []
        self.attr = None

    def search(self, module_or_dir, not_found_error=True):
        # from IPython import embed
        # embed()

        if self.found is not True:
            if type(module_or_dir) == str:
                exec(module_or_dir)
                module = eval(module_or_dir.split(" ")[-1])
            else:
                module = module_or_dir

            attr = None
            self.add_to_history(module)

            if not_found_error is False:
                try:
                    if hasattr(module, self.search_name) is True:
                        attr = getattr(module, self.search_name)
                except ModuleNotFoundError:  # to enable training without torch & tf installation
                    pass
            else:
                if hasattr(module, self.search_name) is True:
                    attr = getattr(module, self.search_name)

            if attr is not None:
                self.found = True
                self.attr = attr

            return attr

    def add_to_history(self, module):
        self.searched.append(module.__name__)

    def end(self):
        if self.found is False:
            print("WARNING: function %s was not found in %s" % (self.search_name, self.searched))
        else:
            return self.attr


def search_function(module_dir, name):
    if hasattr(module_dir, name) is True:
        return getattr(module_dir, name)
    else:
        print("WARNING: function %s was not found in %s" % (name, module_dir.__name__))


def config(args=None):
    import argparse
    from shutil import copytree
    from datetime import datetime

    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--no-logging', dest='logging', action='store_false')
    parser.add_argument('-log_dir', '--log_dir', default='./', dest='log_dir', type=str)
    args = parser.parse_args(args)

    root_dir = expandvars("$WORK")

    if is_interactive() is True:
        args.log_dir = join(root_dir, 'interactive', "training_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    print("\noutput folder\n", args.log_dir)

    os.makedirs(args.log_dir, exist_ok=True)

    if not exists(args.log_dir):
        os.makedirs(args.log_dir + "/")

    repo_dir = join(dirname(realpath(__file__)), '..')

    if args.logging is True:
        copytree(join(repo_dir, 'models', 'tf'), join(args.log_dir, 'bkp', 'tf'), dirs_exist_ok=True)
        copytree(join(repo_dir, 'models', 'torch'), join(args.log_dir, 'bkp', 'torch'), dirs_exist_ok=True)

    return args


def gpu_avail():
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())


def get_current_path():
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    return dirname(abspath(filename))


def create_dir_path(log_dir='./', name='', type=''):
    '''creates new directory of format:
    self.log_dir + / + name + type
    and return directory path
    '''
    import os
    if type != '':
        type = "_" + type
    dir = log_dir + "/" + name + type
    os.makedirs(dir, exist_ok=True)
    return dir


def create_dir(*dirs):
    path = join(*dirs)
    os.makedirs(path, exist_ok=True)
    return path


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return obj.decode('utf-8')
        else:
            return super(NumpyEncoder, self).default(obj)
