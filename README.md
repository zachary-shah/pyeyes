# pyeyes

## Installation
You can create the relevant conda environment using mamba (you can use conda but it will be very slow):
```
mamba env create -n pyeyes --file env.yml
```

Activate the installed environment:
```
mamba activate pyeyes
```

Then install the mr_recon library by moving to the mr_recon directory and running
```
pip install -e ./
```