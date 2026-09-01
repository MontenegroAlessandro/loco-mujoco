# Mind Your Steps: A General Learning Framework for Accurate Humanoid Foothold Tracking

<p align="center">
  <img width="90%" src="assets/new-overview.svg">
</p>

<p align="center"><b>Robotics: Science and Systems (RSS) 2026</b></p>

<p align="center">
  Alessandro Montenegro<sup>1</sup>, Shihao Li<sup>2</sup>, Puze Liu<sup>2†</sup>, Alberto Maria Metelli<sup>1</sup>, Jan Peters<sup>3–6</sup>
</p>

<p align="center">
  <sup>1</sup> Politecnico di Milano &nbsp;&nbsp;
  <sup>2</sup> Tongji University &nbsp;&nbsp;
  <sup>3</sup> Technische Universität Darmstadt &nbsp;&nbsp;
  <sup>4</sup> German Research Center for Artificial Intelligence (DFKI) &nbsp;&nbsp;
  <sup>5</sup> hessian.AI &nbsp;&nbsp;
  <sup>6</sup> Robotics Institute Germany (RIG)
  <br>
  <sup>†</sup> Corresponding Author
</p>

<p align="center">
  <a href="https://montenegroalessandro.github.io/mind-your-steps/">🌐 Website</a> &nbsp;|&nbsp;
  <a href="https://arxiv.org/pdf/2606.08253">📄 Paper</a> &nbsp;|&nbsp;
  <a href="https://youtu.be/mHInn_y-JXs">▶️ Video</a> &nbsp;|&nbsp;
  <a href="experiments/humanoid_foot_placement">💻 Code</a>
</p>

<p align="center">
  <a href="https://youtu.be/mHInn_y-JXs">
    <img width="70%" src="https://img.youtube.com/vi/mHInn_y-JXs/maxresdefault.jpg" alt="Mind Your Steps video">
  </a>
</p>

The training, evaluation, and deployment (sim2sim and sim2real) code for the foot-placement
policies presented in the paper lives in
[`experiments/humanoid_foot_placement`](experiments/humanoid_foot_placement), built on top of
the LocoMuJoCo framework described below.

### Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{montenegro2026mind,
  title     = {Mind Your Steps: A General Learning Framework for Accurate Humanoid Foothold Tracking},
  author    = {Montenegro, Alessandro and Li, Shihao and Liu, Puze and Metelli, Alberto Maria and Peters, Jan},
  booktitle = {Robotics: Science and Systems},
  year      = {2026}
}
```

---

# LocoMuJoCo

**LocoMuJoCo** is an **imitation learning benchmark** specifically designed for **whole-body control**.  
It features a diverse set of environments, including **quadrupeds**, **humanoids**, and **(musculo-)skeletal human models**,
each provided with comprehensive datasets (over 22,000 samples per humanoid).

Although primarily focused on imitation learning, LocoMuJoCo also supports custom reward function classes,  
making it suitable for pure reinforcement learning as well.

## Installation

[//]: # (You have the choice to install the latest release via PyPI by running )

[//]: # ()
[//]: # ()
[//]: # (```bash)

[//]: # ()
[//]: # (pip install loco-mujoco )

[//]: # ()
[//]: # (```)

Clone this repo and do an editable installation:

```bash
cd loco-mujoco
pip install -e . 
```

By default, both will install the CPU-version of Jax. If you want to use Jax on the GPU, you need to install the following:

```bash
pip install jax["cuda12"]
````

> [!NOTE]
> If you want to run the **MyoSkeleton** environment, you need to additionally run
> `loco-mujoco-myomodel-init` to accept the license and download the model.


### Datasets

LocoMuJoCo provides three sources of motion capture (mocap) data for humanoid environments: default (provided by us), LAFAN1, and AMASS. The first two datasets
are available on the [LocoMujoCo HuggingFace dataset repository](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets)
and will downloaded and cached automatically for you. AMASS needs to be downloaded and installed separately due to
their licensing. See [here](loco_mujoco/smpl) for more information about the installation.

This is how you can visualize the datasets:

```python
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf, DefaultDatasetConf, AMASSDatasetConf


# # example --> you can add as many datasets as you want in the lists!
env = ImitationFactory.make("UnitreeH1",
                            default_dataset_conf=DefaultDatasetConf(["squat"]),
                            lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4", "walk1_subject1"]),
                            # if SMPL and AMASS are installed, you can use the following:
                            #amass_dataset_conf=AMASSDatasetConf(["DanceDB/DanceDB/20120911_TheodorosSourmelis/Capoeira_Theodoros_v2_C3D_poses"])
                            )

env.play_trajectory(n_episodes=3, n_steps_per_episode=500, render=True)
```

#### Speeding up Dataset Loading
LocoMuJoCo only stores datasets with joint positions and velocities to save memory. All other attributes are calculated 
using forward kinematics upon loading. If you want to speed up the dataset loading, you can define caches for the datasets. This will
store the forward kinematics results in a cache file, which will be loaded on the next run: 

```bash
loco-mujoco-set-all-caches --path <path to cache>
```

For instance, you could run:
```bash
loco-mujoco-set-all-caches --path "$HOME/.loco-mujoco-caches"
````

## LocoMuJoCo Citation
```
@inproceedings{alhafez2023b,
title={LocoMuJoCo: A Comprehensive Imitation Learning Benchmark for Locomotion},
author={Firas Al-Hafez and Guoping Zhao and Jan Peters and Davide Tateo},
booktitle={6th Robot Learning Workshop, NeurIPS},
year={2023}
}
```




