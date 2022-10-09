# HUNTERPlus


## Quick Start Guide
To run the COSCO framework, install required packages using
```bash
python3 install.py
```
To run the code with the required scheduler, modify line 106 of `main.py` to one of the several options including LRMMTR, RF, RL, RM, Random, RLRMMTR, TMCR, TMMR, TMMTR, GA, GOBI.
```python
scheduler = CNNScheduler('energy_latency_'+str(HOSTS))
```

To run the simulator, use the following command
```bash
python3 main.py
```
```
The GGCN, BiGGCN, and CNN Models are all located in \scheduler\BaGTI\src\models.py
They are known as GRU128, BiGRU128, and CNN

scheduler\BaGTI\src\gnn.py has the backward gated unit.
scheduler\BaGTI\src\opt.py has the optimization function for CNN.

The schedulers are CNN_Normalised and GOBI.
```
