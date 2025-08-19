
How to run: 
1. install all requirements by `pip install -r requirements.txt`
2. Run `python experiments.py --runs 1 --scenarios baseline pv_ramp load_step sensor_fault grid_sag line_outage sensor_bias noise_burst`. use python/python3
3. Run `python analysis.py`

Notes: 
* you can do multiple runs by changing the --runs parameter, and add scenarios to core and can run those as well.
* results will be stored in mentioned directories. 
