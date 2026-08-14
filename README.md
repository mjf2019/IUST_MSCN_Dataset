# IUST_MSCN_Dataset

![Project Logo](images/project-logo.png)

## Introduction

The **Multiscale Computer Network Dataset (MSCN)** is a multiclass network traffic dataset designed to investigate the impact of different levels of network congestion on traffic characteristics and network traffic classification.

The dataset currently contains five traffic classes:

* HTTP
* VIDEO
* SSH
* SFTP
* SMTP

The dataset is generated using the **GNS3 network emulator**, where network traffic is captured as PCAP files under controlled network conditions. Different congestion levels are introduced by generating additional UDP traffic using the **iPerf** tool. The resulting traffic is collected and converted into flow-level data for machine learning and network traffic analysis.

The dataset is particularly designed for studying **congestion-induced changes in network traffic**, including their impact on delay, jitter, packet loss, throughput, and flow-level features.

---

## Network Architecture

The network consists of four local networks interconnected through three routers. The architecture includes host-to-switch, switch-to-gateway, and gateway-to-gateway connections, forming a multi-hop network path containing both switches and routers.

![Network Architecture](images/Dataset_Arch.png)

The following communication paths are used to generate application traffic:

```text
192.168.3.110 -> 192.168.2.110
192.168.4.110 -> 192.168.1.110
```

Different application services are generated over these paths to produce the five traffic classes included in the dataset.

The network is subjected to controlled additional traffic using iPerf. Increasing the offered load causes the network to move toward higher utilization and congestion, resulting in changes in network performance characteristics such as delay, jitter, packet loss, and throughput.

---

## Congestion Generation

To generate reproducible congestion conditions, the network path is modeled as a sequence of finite-buffer processing nodes. Each switch or router has a specific processing capacity, and the node with the lowest service capacity acts as the **bottleneck** of the path.

Instead of selecting the iPerf traffic rate arbitrarily, the required traffic rate is determined according to the target utilization of the bottleneck node and the amount of background traffic already present in the network.

The required iPerf traffic rate is calculated as:

[
b_{\mathrm{iPerf}}(t)=
\max\left(
0,,
\rho_{\mathrm{target}}(t)\mu_{\mathrm{bottleneck}}
-\lambda_{\mathrm{background}}(t)
\right)
\left(L_{\mathrm{packet}}+H_{\mathrm{UDP}}\right)
]

where ( \rho_{\mathrm{target}} ) represents the intended utilization level, ( \mu_{\mathrm{bottleneck}} ) is the service capacity of the bottleneck node, and ( \lambda_{\mathrm{background}} ) represents the existing background traffic.

This approach allows the amount of iPerf traffic required to produce a specific congestion condition to be determined systematically.

Three congestion levels are considered:

| Level   | Congestion |                         Target Utilization |
| ------- | ---------- | -----------------------------------------: |
| Level 1 | Low        |        ( \rho_{\mathrm{target}} \leq 0.6 ) |
| Level 2 | Medium     | (0.35 < \rho_{\mathrm{target}} \leq 0.875) |
| Level 3 | High       |         ( \rho_{\mathrm{target}} > 0.875 ) |

The resulting network delay is also monitored during the experiments to characterize the actual effect of each congestion level.

The iPerf traffic is applied consistently across the different application services so that the effect of congestion can be evaluated independently of differences in the amount of additional load. Consequently, variations in delay, jitter, packet loss, and other traffic characteristics can be analyzed as a function of the network congestion condition.

To make the generated traffic more representative of dynamic network environments, the amount and duration of additional iPerf traffic can also be varied within the predefined congestion conditions.

---

## Pcap to NetFlow

After generating the network traffic, the resulting PCAP files are converted into flow-level records using **[Argus](https://openargus.org/)**.

Argus is used to extract the available flow-level features from the captured traffic. Different flow-generation time scales are also considered, ranging from **0.001 seconds to 60 seconds**.

Using different time scales changes the granularity of the generated NetFlow records and consequently changes the number and characteristics of the resulting flows.

The resulting data is organized according to traffic class and flow-generation scale.

---

## Preprocessing

The generated NetFlow files are converted into CSV files during the preprocessing stage.

The main preprocessing operations include:

* Handling missing values
* Removing columns containing insufficient information
* One-hot encoding of non-numeric features
* Feature scaling using Min-Max normalization

The preprocessing configuration is specified in `project_conf.yaml`.

For example:

```yaml
original_dataset_path: "original_dataset\\scale_30\\SFTP\\flow\\all\\"
preprossed_dataset_path: "preprocessed_dataset\\scale_30\\SFTP\\all\\"
```

The preprocessing can be executed using:

```bash
python preprocess_netflow.py
```

Because one-hot encoding is applied independently to the generated files, the resulting CSV files may initially contain different feature sets.

---

## Standardization and Dataset Integration

To obtain a consistent feature space across the generated files, the preprocessed datasets are integrated and standardized.

First, run:

```bash
python standard_pre_data_integration.py
```

Then run:

```bash
python dataset_standard.py
```

Depending on the NetFlow generation scale, the final standardized datasets contain approximately **43 to 56 features**.

The resulting `standard_dataset` is intended for analyzing individual traffic classes under different congestion conditions.

For experiments involving multiple traffic classes and congestion levels, the standardized datasets are combined using:

```bash
python dataset_merger.py
```

The resulting `merged_standard_dataset` is intended for evaluating multiclass and multi-label traffic classification models under different congestion conditions.

---

## Dataset Organization

The dataset generation and processing pipeline can be summarized as follows:

```text
GNS3 Network
     |
     v
Application Traffic
(HTTP, VIDEO, SSH, SFTP, SMTP)
     |
     +---- iPerf Controlled Load
     |
     v
Different Congestion Levels
(Low / Medium / High)
     |
     v
PCAP Capture
     |
     v
Argus
     |
     v
NetFlow
     |
     v
Preprocessing
     |
     v
Standardization
     |
     +--------------------+
     |                    |
     v                    v
standard_dataset   merged_standard_dataset
     |                    |
     v                    v
Per-Class Analysis   Multi-Class /
                     Multi-Label Analysis
```

---

## Research Applications

The dataset can be used to investigate:

* Network traffic classification under different congestion conditions
* Congestion-induced distribution shift
* Robustness of machine learning models to changing network conditions
* Impact of congestion on flow-level traffic characteristics
* Multiclass and multi-label traffic classification
* Adaptation to dynamically changing network environments
* Concept drift in network traffic classification

---

## Methods

Each evaluated method is organized in a separate folder at the root of the repository. The corresponding folder contains the implementation and required files for that method.

The repository currently includes:

* **CDR-MLC** — located in the `CDR-MLC/` folder.
* **AMCAL** — located in the `AMCAL/` folder.

To reproduce the results for a specific method, navigate to its corresponding folder and follow the instructions provided in its README or execution scripts.
