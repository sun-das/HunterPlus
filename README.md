<h1 align="center">HunterPlus: AI based energy-efficient task scheduling for cloud–fog computing environments</h1>
<div align="center">
  <a>
    <img src="https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg" alt="Python 3.7, 3.8">
  </a>
   <a>
    <img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fimperial-qore%2FCOSCO&count_bg=%23FFC401&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false" alt="Hits">
  </a>
   <a href="https://github.com/imperial-qore/COSCO/actions">
    <img src="https://github.com/imperial-qore/SimpleFogSim/workflows/DeFog-Benchmarks/badge.svg" alt="Actions Status">
  </a>
  </a>
   <a href="https://doi.org/10.5281/zenodo.4897944">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4897944.svg" alt="Zenodo">
  </a>
 <br>
   <a>
    <img src="https://img.shields.io/docker/pulls/shreshthtuli/yolo?label=docker%20pulls%3A%20yolo" alt="Docker pulls yolo">
  </a>
   <a>
    <img src="https://img.shields.io/docker/pulls/shreshthtuli/pocketsphinx?label=docker%20pulls%3A%20pocketsphinx" alt="Docker pulls pocketsphinx">
  </a>
   <a>
    <img src="https://img.shields.io/docker/pulls/shreshthtuli/aeneas?label=docker%20pulls%3A%20aeneas" alt="Docker pulls aeneas">
  </a>
 <br>
   <a href="https://gitpod.io/#https://github.com/imperial-qore/COSCO/">
    <img src="https://gitpod.io/button/open-in-gitpod.svg" alt="Open in gitpod">
  </a>
</div>

COSCO is an AI based coupled-simulation and container orchestration framework for integrated Edge, Fog and Cloud Computing Environments. It's a simple python based software solution, where academics or practitioners can develop, simulate, test and deploy their scheduling policies. Further, this repo presents a novel gradient-based optimization strategy using deep neural networks as surrogate functions and co-simulations to facilitate decision making. A tutorial of the COSCO framework was presented at the International Conference of Performance Engineering (ICPE) 2022. Recording available [here](https://youtu.be/osjpaNmkm_w).

<img src="https://github.com/imperial-qore/COSCO/blob/master/wiki/COSCO.jpg" width="900" align="middle">


## Advantages of COSCO
1. Hassle free development of AI based scheduling algorithms in integrated edge, fog and cloud infrastructures.
2. Provides seamless integration of scheduling policies with simulated back-end for enhanced decision making.
3. Supports container migration physical deployments (not supported by other frameworks) using CRIU utility.
4. Multiple deployment support as per needs of the developers. (Vagrant VM testbed, VLAN Fog environment, Cloud based deployment using Azure/AWS/OpenStack)
5. Equipped with a smart real-time graph generation of utilization metrics using InfluxDB and Grafana.
6. Real time metrics monitoring, logging and consolidated graph generation using custom Stats logger.

The basic architecture of COSCO has two main packages: <br>
**Simulator:** It's a discrete event simulator and runs in a standalone system. <br>
**Framework:** It’s a kind of tool to test the scheduling algorithms in a physical(real time) fog/cloud environment with real world applications.

Supported workloads: (Simulator) [Bitbrains](http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains) and [Azure2017/2019](https://github.com/Azure/AzurePublicDataset); (Framework) [DeFog](https://github.com/qub-blesson/DeFog) and [AIoTBench](https://www.benchcouncil.org/aibench/aiotbench/index.html).

Our main COSCO work uses the Bitbrains and DeFog workloads. An extended work, MCDS (see `workflow` branch), accepted in IEEE TPDS uses scientific workflows. Check [paper](https://arxiv.org/abs/2112.07269) and [code](https://github.com/imperial-qore/COSCO/tree/workflow).

## Abstract
Cloud computing is a mainstay of modern technology, offering cost-effective and scalable solutions to a variety of different problems. The massive shift of organization resource needs from local systems to cloud-based systems has greatly increased the costs incurred by cloud providers in expanding, maintaining, and supplying server, storage, network, and processing hardware. Due to the large scale at which cloud providers operate, even small performance degradation issues can cause energy or resource usage costs to rise dramatically. One way in which cloud providers may improve cost reduction is by reducing energy consumption. The use of intelligent task-scheduling algorithms to allocate user-deployed jobs to servers can reduce the amount of energy consumed. Conventional task scheduling algorithms involve both heuristic and metaheuristic methods. Recently, the application of Artificial Intelligence (AI) to optimize task scheduling has seen significant progress, including the Gated Graph Convolution Network (GGCN). This paper proposes a new approach called HunterPlus which examine the effect of extending the GGCN’s Gated Recurrent Unit to a Bidirectional Gated Recurrent Unit. The paper also studies the utilization of Convolutional Neural Networks (CNNs) in optimizing cloud–fog task scheduling. Experimental results show that the CNN scheduler outperforms the GGCN-based models in both energy consumption per task and job completion rate metrics by at least 17 and 10.4 percent, respectively.


## Quick Start Guide
To run the this framework, install required packages using
```bash
python3 install.py
```
To run the code with the required scheduler, modify line 106 of `main.py` to one of the several options including LRMMTR, RF, RL, RM, Random, RLRMMTR, TMCR, TMMR, TMMTR, GA, GOBI.
```python
scheduler = GOBIScheduler('energy_latency_'+str(HOSTS))
```

To run the simulator, use the following command
```bash
python3 main.py
```


## Links
| --- | --- |
| **Paper** | https://www.sciencedirect.com/science/article/pii/S2542660522001482 |
| **Contact**| s.iftikhar@qmul.ac.uk |

## Cite this work
Our work is published in Internet of Things journal. Cite using the following bibtex entry.
```bibtex
@article{iftikhar2023hunterplus,
  title={HunterPlus: AI based energy-efficient task scheduling for cloud--fog computing environments},
  author={Iftikhar, Sundas and Ahmad, Mirza Mohammad Mufleh and Tuli, Shreshth and Chowdhury, Deepraj and Xu, Minxian and Gill, Sukhpal Singh and Uhlig, Steve},
  journal={Internet of Things},
  volume={21},
  pages={100667},
  year={2023},
  publisher={Elsevier}
}
```
