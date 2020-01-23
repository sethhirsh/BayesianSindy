# BayesianSindy

Working repo for the PNNL MARS project "Probabilistic Model Discovery with Uncertainty Quantification"

## Creating the environment

With the `.yml` file it suffices to say

```bash
$CONDA env create -f environment.yml
```

where `$CONDA` is the `conda` (Anaconda/Miniconda) binary.  If the Anaconda binaries were added to `$PATH`, then this is simply `conda`.

## Activating and deactivating the environment

If it hasn't been done before, it is necessary to set up the shell to activate and deactivate environments.  For that, do

```bash
$CONDA init
$CONDA config --set auto_activate_base false
```

and restart the shell.  The last line will prevent the `base` environment from being loaded when the shell is launched (so that you have access from the shell to whatever Python you had first on your path, e.g., macOS's Python).

After restarting the shell, `which conda` should *definitely* point to the `conda` binary, and the remaining instructions assume that, if that's not the case, let David know!

Once the shell is set up, activating is done by

```bash
CONDA activate mars
```

which will add a `(mars)` prefix to the prompt.  After this, `which python` and  `which jupyter` should point to the `mars` environment's Python and Jupyter, respectively.  You can then launch Jupyter the usual way: `jupyter lab`.

To deactivate the environment, do

```bash
CONDA deactivate
```

After this, `which python` should point to the whatever other Python you had on your path with first priority.
