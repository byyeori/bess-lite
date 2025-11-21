# Bess-Lite

Lightweight forecasting model for Battery Energy Storage Systems (BESS).

## Overview
Renewable energy sources such as solar and wind introduce strong variability and unpredictability into modern power grids. To mitigate this, Energy Storage Systems (ESS) store surplus energy and stabilize supply when fluctuations occur. However, real-world ESS controllers often run on resource-constrained hardware, requiring models that are not only accurate but also lightweight and energy-efficient. Traditional deep learning approaches—LSTM-based solar forecasting, autoencoders, deep reinforcement learning for energy brokerage, and Q-learning for battery management—achieve high accuracy but are too heavy for deployment in embedded ESS controllers. Their computational cost, memory usage, and slow inference make them impractical for real-time, large-scale ESS operations. `bess-lite` aims to address this gap by developing a lightweight, efficient forecasting model optimized for ESS operation. The model predicts key signals such as PV generation, wind output, and load/net-load while maintaining fast inference speed and minimal resource usage. This makes it suitable for real-time ESS control, embedded devices, and large-scale ESS deployments.

## Goals
- Develop a lightweight deep learning architecture tailored for ESS operation.
- Apply pruning, quantization, and knowledge distillation to reduce model size.
- Improve forecasting accuracy using real-world operational data.
- Evaluate practical deployability using metrics such as inference latency, memory footprint, and energy efficiency.

This project provides a compact yet practical forecasting model designed for real deployment in ESS controllers and field devices.
