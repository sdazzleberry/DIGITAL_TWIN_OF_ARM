# DIGITAL_TWIN_OF_ARM

A MuJoCo-based digital twin for a cycloidal arm with wearable IMU-driven teleoperation and learned state estimation.

This project combines:
- simulation of a robotic arm and virtual IMUs
- realistic sensor corruption for domain randomization
- live teleoperation from wearable IMU + EMG signals
- a causal temporal convolutional network (TCN) for joint-state estimation

## Why this project

Estimating arm motion from wearable sensors is hard because real IMU data is noisy, biased, and affected by sensor placement. Instead of training on ideal signals, this project builds a realistic synthetic pipeline:
- generate clean motion and IMU data in simulation
- corrupt those signals to mimic real hardware behavior
- train a model on those realistic observations
- use the trained model for teleoperation or estimation in a digital twin

This makes the learning problem closer to deployment conditions while keeping the dataset controllable and reproducible.

## Repository structure

```text
.
├── sim_imu_dataset.py      # generate simulated IMU dataset
├── imu_augment.py          # realistic sensor corruption / domain randomization
├── imu_teleop.py           # wearable IMU + EMG teleoperation loop
├── train_tcn.py            # causal TCN training pipeline
├── sim_comms.py            # simulator communication layer
├── scene.xml               # base MuJoCo scene
├── scene_with_imus.xml     # scene with virtual IMUs
├── cycloidal_arm.xml       # arm model
├── data/                   # generated dataset files
├── models/                 # trained model outputs
├── experiments/            # experiment runs
├── meshes/                 # assets
└── README.md
