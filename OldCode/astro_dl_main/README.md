# astro_dl
Pipeline for training deep learning models using TensorFlow, PyTorch or PyTorch_Geometric to reconstruct data measured by astroparticle detectors.

The current release is tested only with Python 3.9.12, CUDA 11.3, TensorFlow 2.9.1, Torch 1.12.1, and torch_geometric 2.1.0.
The following cuDNN versions are used:
Torch: cuDNN 8302  # torch.backends.cudnn.version()
TensorFlow: cuDNN version 8201  # Currently no more recent version exist @ conda-forge for CUDA 11.3


# Installation
There are several ways to install `astro_dl`.

### Conda image
The simplest way and **recommendation** is to use the conda image add `install/conda.yml`

So first install miniconda (**PYTHON 3.9**) if you use the default installer:
using, e.g., https://docs.conda.io/en/latest/miniconda.html#linux-installers

Make sure to set the path of anaconda to a reasonable directory (**SO NOT YOUR HOME DIRECTORY**), e.g., create a sofware folder at you $WORK directory.

Simply create the environment using:

```bash
conda env create -f install/astro_dl.yml
```

Afterwards, just the repository has to be linked to your \$PYTHONPATH, the \$LD_LIBRARY_PATH for CUDA and cuDNN support, and set your \$ASTRODLENV.
Note, if you used a ```-p``` or ```--prefix``` during conda create activate your environment using ```conda activate YOURPREFIX```.

```bash
cd install/
source make_env.sh
```
## SWGO

For enabling an easy interface between astro_dl and the SWGO data, pyswgo should be installed.

### Fast install
Alternatively use the merged installer at:
```bash
conda env create -f swgo/pyswgo_astrodl_merged.yml
```
Note that this installer might be not up to date (updated Mid Jan 2023).

### Full install
For the full installation we simply queue the astro_dl and pyswgo installation. Therefore, git is needed:
Install astro_dl without explicit version requirements using:
```bash
conda env create -f install/astro_dl_only_cuda_dep_frozen.yml
```

Then install git
```bash
conda install -c conda-forge git
```

Afterwards, pull the repository:
```bash
git clone https://gitlab.com/swgo-collaboration/irf-production.git
```

For installing the needed software:
Firstly, activate your environment:
```bash
conda activate astro_dl
```

Secondly, install the software.
```bash
cd irf-production
conda env update --file environment.yaml --prune --prefix full_path_to_your_astro_dl_env
```
Do not forget to set the ```--prefix``` option if it were used to install ```astro_dl```. Or if you did not use the ```--prefix``` but gave the path manually to conda when asked to be installed at a different location. Then in the latter case also you should give the full path of the astro_dl env you want to update with the new environment.yaml file. Note, this will may update packages of astro_dl. Thus, the version of single packages might change after the installation.


Thirdly, install the rest of the pyswgo software:
```bash
conda activate astrodl
pip install -e '.[all]'
```

Afterwards locate to you astro_dl installation and set up you astro_dl env

```bash
cd install/
source make_env.sh
```


### Latex support
Install texlive for latex rendering in matplotlib. See https://www.tug.org/texlive/quickinstall.html for detailed installation instructions.
```bash
    cd /tmp # working directory of your choice
    wget https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz # or curl instead of wget
    zcat install-tl-unx.tar.gz | tar xf -
```
```bash
    mkdir $HOME/latex
    cd install-tl-$YOURVERSION
    perl ./install-tl --no-interaction --texdir $HOME/latex # as root or with writable destination
    Finally, prepend /usr/local/texlive/YYYY/bin/PLATFORM to your PATH,
    e.g., /usr/local/texlive/2022/bin/x86_64-linux
```

#### Comments on PyTorch and TensorFlow CUDA CuDNN installations
Both software packages need CUDA. cuda_toolkit
Up to now, torch ships directly with a 

### Docker container
As another alternative we provide the software as a docker container.
Simply use our Docker files located at ```install/docker``` to use the framework as docker container.


# TensorFlow, PyTorch, and PyTorch_Geometric
The framework supports the training of models designed using TensorFlow (Keras), PyTorch, and PyTorch_Geometric.
The training of TF models relies on the Keras API.
Please design models as follows:
Add them into the respective folder: experiment/models
Order the inputs using keys.
The TF.datasets (recommended to use / very efficient) class relies on a python dictionary. Thus, for each input / output the respective key is required. Furthermore, the full data set is of the form (input, output). Where input, i.e. "feat", is
```
feat={"ct1": tf.Tensor([[]]), "ct2": tf.Tensor([[]])}, labels =  {"primary": tf.Tensor([]), "energy": tf.Tensor([[])}
```

## Philosophy
The framework targets to be a simple solution for training models designed using Keras, PyTorch, or PyTorch_Geometric in a homogeneous manner without re-writing the code for each physics experiment.

## Data sets
The data should be created as HDF5 file and loaded as an instance of the ```my_data = DataContainer(.....)``` class.
The class enables a simple conversion to:
- TF Datasets by calling ```my_data.tf()```.
- PyTorch Dataset by calling ```my_data.torch()```.
- List of PyTorch Geometric Data instances by calling ```my_data.torch_geometric()```.
--> The actual dataset can be accessed by calling the dataset, e.g., my_data().
In the beginning, it will simply return the numpy values. After you build your TF/Torch/PyG datasets you can simply acess your TF/Torch/PyG data by calling the dataset (```my_data.to_np()```). You can also convert the datasets back to numpy arrays using ```my_data.to_np()```.


## Pre-processing
The pre-processing should be performed using the respective features of the underlying dataset classes, e.g., using tf.Datasets.

## Graphs: Handling particle detectors as point clouds
The framework handles incoming data as point clouds, i.e., no graph is defined a priori.
Define the graph when building the PyG dataset, e.g., using ```DataContainer.torch_geometric(graph_transform=T.KNNGraph(k=6))```

### Important to know
For the correct batching in PyG, the following namespace has to be used.
The features can have an arbitrary name: e.g., the photoelectrons measured by H.E.S.S. in the respective telescopes can be saved as "ct1", "ct2", ....
The position of the points in the point clouds should have the namespace "pos_". Thus, the pixel positions of the H.E.S.S. would be "pos_ct1", "pos_ct2", ....
Using the graph_transform, a graph is created out of the point cloud. The respective sparse adjacency (edge_index) matrix is stored as "ei_pos_" (ei = edge index). Thus for the H.E.S.S. case the created adjacency matrices are "ei_pos_ct1", "ei_pos_ct2", ...
When using the DataLoader, in addition, for each input, a batch tensor is created (describing which node/feature belongs to which sample in the batch), which is needed to perform pooling.

## Model design
The framework supports model design in a Keras-like fashion. Thus, inputs and outputs should be created as a python dictionary to enable straightforward multi-task and multi-input training.

## Trainer
- The trainer class should simplify the training of models and generalizes TF, and torch training loops. Furthermore, it supports ```tensorbard``` for helpful visualizations and tracking of the training.
- It requires a model and data
- *ToDo: enable simple hyperparameter searches without flags*

## Framework design
Scheme of the underlying framework
- Opening of HDF5 file --> casting to NumPy array (at the moment loaded into RAM)
- Dividing into training, validation, and test set. Each of these data sets is an instance of the ```DataContainer``` class, which supports conversion into tf.data.Datasets, torch.data.Dataset, and list of torch_geometric.data.Data objects.
- Conversion to respective format (These are only references, so no loading into RAM)
- Batching using tf.datasets.Data.batch, torch.data.loader, or pytorch_geometric.data.loader (usually inside Trainer class)
- Batches of data should be of type direct (for PyTorch_Geometric, we call model.batch2dict2device(batch))


## TASKS
For the management of the learning tasks, astro_dl uses ```tasks```. The tasks are used manage your multi-task training to set meaningful losses (and handle the access to TF/torch), handle the metrics for monitoring the training and evaluating the models, and error catching/solving.

Currently three learning tasks are available:
- ```models.tasks.Regression```
- ```models.tasks.Classification```
- ```models.tasks.VectorRegression``` (for reconstructing multi-dimensional objetcs, i.e., vector, e.g., shower core, position). For reconstructing angular-related quantities (axis vector defined using a unit sphere) set type="angular" for euclidean geometries like shower cores use type="euclidean".
- ToDo implement segmentation!

In the following we define the tasks for our multi-task-learning challenge: a classification into 10 classes, the energy and shower core reconstruction, and the reconstruction of the arrival direction (shower axis).
Note, reconstructing angular is not recommended due to the poles at zenith=0 and the periodicity of phi.

```
  TASKS = {'particle_type': Classification(num_classes=10), 'energy': Regression(unit="TeV"), 'arrival_direction': VectorRegression("angular", unit="deg"), 'shower_core': VectorRegression("euclidean", unit="m")}
```
## Evaluation
Model evaluation is performed using the ```Evaluator``` class that utilizes (meaningful) performance **metrics** (see below) that are assigned and managed via the learning **task** (see above).
For running the evaluation, which includes the prediction, simply run:
```
evaluation = evaluate.Evaluator(my_aiact.model, test_datasets, TASKS, log_dir=CONFIG.log_dir)
evaluation.evaluate()
```
For studying the perforance as a function of another observable make us of the ```observable_dep``` attribute.
E.g., for plotting the model performance as a function of the *energy* in 30 (log_10)bins from 0.01 to 300 TeV do:    
```
evaluation.observable_dep("energy", (0.01, 300, 30), log=True)
```

## METRICS
The performance of models is evaluated using metrics.
A metric is a scalar function that evaluates the performance by returning a single scalar, e.g., resolution, accuracy, or bias are metrics.
As often a single value might under-represent the complexity of a metric and a metric is commonly accompanied by a merit figure, we provide the option to visualize the metric using matplotlib.

For implementing a metric into astro_dl
- the ```metric_fn``` has to be implemented in pure numpy (use numpy operations only and take care of your RAM usage)

For monitoring the metric during training, furthermore, the scalar mapping has to be implemented in:
- TensorFlow (add the metric to models.tf.tf_metrics), for training TF/Keras models
- Torch (add the metric to models.torch.torch_metrics), for training torch/PyG models

Note, the software is looking for the correct name. So please you the **same name** for the TF/torch metric as for the numpy implementation.

For adding a merit figure, simply modify the plot function of the metric.
You can find an example implementation below:

```
class ScatterMetric(RegressionMetric):
    def __init__(self, metric_fn, vis_fn=plt.scatter, xlabel="y_true", ylabel="y_pred", unit=None, name=''):
        super().__init__(metric_fn, vis_fn=vis_fn, xlabel=xlabel, ylabel=ylabel, unit=None, name=name)

    def plot(self, ax, y_true, y_pred, name="", **plt_kwargs):
        results = self(y_true, y_pred)
        label = "%s: %.2f" % (name, results)
        self.vis_fn(y_true, y_pred, axes=ax, label=label, **plt_kwargs)

    def set_ylabel(self, task_name, ax):
        ax.set_ylabel("%s_{pred} / %s" % (task_name, self.unit))

    def set_xlabel(self, task_name, ax):
        ax.set_xlabel("%s_{true} / %s" % (task_name, self.unit))
```
For implementing a correlation metric with accompanied scatter plot simply do:

```
def correlation(x,y)
    return np.corrcoef()

corr_metric = ScatterMetric(correlation)
```


## Job submission using the FAU batch system
Submission of jobs at the FAU GPU cluster *Alex* is managed by the slurm software.
Astro_dl provides a pythonic wrapper for your job submissions.

Simply locate to astro_dl. For running a file ```MYFILE.py``` on a GPU node simply do:
```
python submit.py --f MYFILE.py
```
Make sure that the .bash_profile exists and add these lines to it:
```
# Template .bashrc file to be stored at
# /home/hpc/<GROUP>/<USER>/
# It simply executes the .bashrc-file

source $HOME/.bashrc
```
Note, if you use a custom environment please activate your conda env before or set an $ASTRODLENV variable.

When using ```CONFIG``` of ```astro_dl.tools.utils``` a folder structure will be created in your working directory ($WORK). The output of jobs started in a given experiment folder will be assigned to your experiment and saved into a new folder in your experiment folder.
All results (training configs / TensorBoard files / plots) will be stored at this location (your ```log_dir```).


### Interactive Jobs
Run interactive job on GPU cluster (e.g., via woody or alex).
Consider to use the alex cluster for computationally extensive applications.

```bash
salloc.tinygpu --gres=gpu:1 --time=01:00:00
```

Run interactive job on Alex cluster (e.g., via Alex)
```bash
salloc --gres=gpu:a40:1 --time=01:00:00
```

Connect to a running job and open a shell at the running node:
```bash
srun --jobid=$JOBID --pty /bin/bash
```



# Manual installation (set up your own environment)
Note that the foundation of the installation is Python CUDA and cuDNN.

To install CUDA and cuDNN using conda-forge do

If libxml is not installed, install it, e.g., in a debian environment do:

```bash
apt install libxml2
```
 Then

```bash
conda install -c conda-forge cudatoolkit-dev==11.3.1
conda install -c conda-forge cudnn=8.4.1
 ```
 Alternatively, (for non-developer cuda)
 ```bash
 conda install -c conda-forge cudatoolkit=11.3 cudnn=8.4.1
 ```



### Install TensorFlow (here using cuda 11.3)

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
python3 -m pip install tensorflow
```
To save the variable to you new environment
```bash
conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
```

Find more information here: https://www.tensorflow.org/install/pip

```bash
# Verify install:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Install torch and PyTorch geometric (here using cuda 11.3)
First install PyTorch using;
Attention: Try to use meaningful combinations of cuda and cudnn, especially when you want to run it in parallel with TensorFlow. For example, a reasonable choice could be:
```bash
conda install pytorch=1.12.1=py3.9_cuda11.3_cudnn8.3.2_0 torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

For a more general installation use:
```bash
conda install PyTorch torchvision torchaudio cudatoolkit=11.3 -c PyTorch
```
or find more information here: https://pytorch.org/get-started/locally/

```bash
# Verify install:
python3 -c "import torch; print(torch.cuda.is_available())"
```

In the following install PyG (check that correct PyTorch version was installed)
```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-Conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
```

Find more information here: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

```bash
# Verify install:
python3 -c "import torch_geometric; print(torch_geometric.__version__)"
```


### Additional required packages

Subsequently, install the additional required packages using conda and pip.

```bash
conda install --file install/requirements_conda.txt
```
Then install (dependening on the version CUDA / CPU).
For CUDA support:
```
pip install -r install/requirements_pip.txt
```
else:
```
pip install -r install/requirements_pip_nocuda.txt
```

Install HDF5plugins to enable support of BLOSC filter in h5py files.

```bash
conda install -c conda-forge hdf5plugin
```

Install Tensorboard for training visualizations.

```bash
conda install -c conda-forge tensorboard
```

* Optional: Hexaconv*
Install hexaconv compatible with tf2.X and 1.13.1.
and add hexaconv to your PYTHONPATH

```bash
git clone https://github.com/Napoleongurke/hexaconv.git ../hexaconv
export PYTHONPATH=$PWD/../hexaconv:$PYTHONPATH
```

## Build docker images and conda environments
The conda environments are build using an empty docker environment

For building the images, simply install docker and navigate to the install folder:

```bash
cd install/
```

Then perform the automatic building of docker images and conda environments:

```bash
rm astro_dl.yml
rm astro_dl_nocuda.yml
./build_conda_envs
```
# Errors

## CUDA OUT OF MEMORY ERROR
Note that TF is claiming all GRAM when initializing operations. Thus, be sure that no TF is running if you want to use torch from training your DNNs. The first time TensorFlow operations are initialized is during the building of your metrics!

## CuDNN errors
Check that you running only tensorflow / only torch.
If you import torch before starting training with tensorflow. Tensorflow uses the CuDNN compiled during the torch installation. This can lead to errors!

## Running the test suite

```bash
pytest
flake8 .
```
