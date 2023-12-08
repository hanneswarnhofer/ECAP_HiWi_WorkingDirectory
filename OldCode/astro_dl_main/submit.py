import subprocess
import argparse
import os
from os.path import join, dirname, realpath, exists, expandvars
import time
import distutils
from shutil import copy

parser = argparse.ArgumentParser(description='Submit script for python scripts')

parser.add_argument("--f", action="store", dest='file', default=None, type=str)
parser.add_argument("--n", action="store", dest='n_jobs', default=1, type=int)
parser.add_argument("--m", action="store", dest='name', default="", type=str)
parser.add_argument("--nodes", action="store", dest='nodes', default="1", type=str)
parser.add_argument("--cpus-per-task", action="store", dest='cpus', default="1", type=str)
parser.add_argument("--ntasks-per-node", action="store", dest='tasks', default="1", type=str)
parser.add_argument("--time", action="store", dest='time', default="24:00:00", type=str)
parser.add_argument("--g", action="store", dest='gpu_type', default="a100", type=str)
parsed = parser.parse_args()

assert parsed.file is not None, "No file for submission given. Add it via 'python submit.py --f $FILE'"
assert ".py" in parsed.file, "Submitted file has to be a pyton file"


work_dir = expandvars("$WORK")
env_dir = expandvars("$CONDA_DEFAULT_ENV")

if env_dir == "base":
    env_dir = expandvars("$ASTRODLENV")

if "astro_dl" not in env_dir:
    print("\n - - - - - - - - - - - - - - - - - !!!! WARNING: !!!! - - - - - - - - - - - - - - - - - ")
    print("It looks like you did not activate your astro_dl conda environment.")
    print("Either activate your environment (for default installation: 'conda activate astro_dl'")
    print("Or declare your env as $ASTRODLENV variable in your .bashrc")
    print(" - - - - - - - - - - - - - - - - - !!!! WARNING: !!!! - - - - - - - - - - - - - - - - - \n")

for i in range(parsed.n_jobs):

    if parsed.name == "":
        file_name = parsed.file.split(".py")[0]
    else:
        file_name = parsed.name

    astro_dl_dir, folder = realpath(parsed.file).split("astro_dl")
    folder = folder.split("/")[1]
    astro_dl_dir += "astro_dl"

    out_dir = str(join(work_dir, "jobs", folder))
    file_path = realpath(parsed.file)
    script_dir = dirname(realpath(parsed.file))
    os.makedirs(out_dir, exist_ok=True)

    job_dir = join(out_dir, file_name)
    run_id = 1

    while exists(job_dir + "_job_%i" % run_id):
        run_id += 1

    job_dir += "_job_%i" % run_id
    print("Creating folder.....", job_dir)
    os.makedirs(job_dir)
    model_dir = join(job_dir, "models")
    os.makedirs(model_dir)

    copy(file_path, job_dir)

    try:
        distutils.dir_util.copy_tree(join(script_dir, '../models'), model_dir)
    except distutils.errors.DistutilsFileError:
        try:
            distutils.dir_util.copy_tree(join(script_dir, 'models'), model_dir + "_inplace")
        except distutils.errors.DistutilsFileError:
            pass

    batch_file = join(job_dir, "batch_script.sh")

    with open(batch_file, "w") as f:
        f.writelines('#!/bin/bash -l\n')
        f.writelines("\n#SBATCH --nodes=%s" % parsed.nodes)
        f.writelines("\n#SBATCH --ntasks-per-node=%s" % parsed.tasks)
        f.writelines("\n#SBATCH --cpus-per-task=%s" % parsed.cpus)
        f.writelines("\n#SBATCH --time=%s" % parsed.time)
        f.writelines("\n#SBATCH --gres=gpu:%s:1" % parsed.gpu_type)
        f.writelines("\n#SBATCH --output=output_test.out")
        f.writelines('\n\nconda activate %s\n' % env_dir)
        f.writelines("\npython %s --log_dir %s" % (file_path, job_dir))

    time.sleep(0.1)
    process = subprocess.run(["cd %s && sbatch ./batch_script.sh" % job_dir], capture_output=True, shell=True)
    print(process.stdout.decode())
    time.sleep(1)
