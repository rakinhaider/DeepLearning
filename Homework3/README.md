## CS69000-DPL - (Graduate) Deep Learning

Homework repository for **CS69000-DPL - (Graduate) Deep Learning**.

## Installation

Run following codes on a totally fresh login to `scholar-fe04.rcac.purdue.edu` (can be `fe-05` or `fe-06`).

```bash
module purge
module load anaconda/2020.11-py38
rcac-conda-env create -n DPL-GPU python=3.8 ipython ipykernel
module load use.own
module load conda-env/DPL-GPU-py3.8.5
conda install matplotlib scikit-learn seaborn
conda install pytorch cudatoolkit=10.2 -c pytorch
```

If you make any mistake, you can remove your environment by

```bash
rcac-conda-env delete -n DPL-GPU
```

## Activate

Create a file `DPL-GPU.rc` in the home `~` directory and put following code into it.

```bash
module purge
module load anaconda/2020.11-py38
module load use.own
module load conda-env/DPL-GPU-py3.8.5
echo -e "[DPL-GPU] \033[92mActivated\033[0m"
```

**Every time you login into `scholar` cluster, run `source DPL-GPU.rc` to activate the environment.**

For guys with VSCode which may have python version problem (use cluster default python instead of installed python).

Try use following one instead.

```
module purge
unset PYTHONPATH
module load anaconda/2020.11-py38
module load use.own
module load conda-env/DPL-GPU-py3.8.5
echo -e "[DPL-GPU] \033[92mActivated\033[0m"
```

## Test

Now create a python file `cuda.py` to test GPU supporting.

```python
# All imports.
import torch

# Create a random tensor and upload to GPU.
x = torch.zeros(3, 3)
x = x.to("cuda")
print(x)
```

If you are on the front-ends supporting GPU (`scholar-fe04`, `scholar-fe05`, `scholar-fe06`), you can run directly in the terminal by `python cuda.py`.

If you want to test `sbatch`, then create another submission file `cuda.sb`.

```bash
#!/bin/bash
#SBATCH -A gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --time=30:00
#SBATCH --job-name TestGPU
#SBATCH --output TestGPU.out
#SBATCH --error TestGPU.err


# Run python file.
# You can run directly as long as you are in the activated environment.
# Otherwise, you need add
# ```bash
# module load anaconda/2020.11-py38
# module load use.own
# module load conda-env/DPL-GPU-py3.8.5
# ```
# ahead of this line.
python ./cuda.py
```

Then, run `sbatch cuda.sb`.

## Additional Packages

You may need to install additional packages with class going on.

- Homework 2

  You need to install `torchvision` and `seaborn`.

  ```bash
  conda install torchvision -c pytorch
  conda install seaborn
  ```

