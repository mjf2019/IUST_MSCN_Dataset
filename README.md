# IUST_MSCN_Dataset
![Project Logo](images/project-logo.png)

## Introduction
The multiscale computer network dataset (MSCN) is a multiclass traffic at different perturbation levels.
This data set is designed to show different levels of congestion in the network and currently includes 5 different behavior classes: HTTP, VIDEO, SSH, SFTP, SMTP.
GNS3 emulator is used to create and save network traffic (PCAP). The general architecture for building the first version of the dataset is given in the figure below.

## Network Architecture
In the design of the network architecture, host-switch, switch-gateway, and gateway-gateway connections have been implemented. This network consists of four local networks connected to each other through three routers. The iperf tool is used to add overhead to the network. Adding overhead causes congestion, which affects network performance metrics such as delay, jitter, loss, and throughput. These changes vary depending on the volume of load exchanged within the network.

![Project Logo](images/GNS3_Arch.png)

To create different levels of network performance, hosts are used as follows:

    192.168.3.110 -> 192.168.2.110
    192.168.4.110 -> 192.168.1.110

Additionally, to simulate real-world traffic, both the amount of overhead added and the duration for which the overhead is applied by iperf are chosen randomly. To increase the performance level, the minimum overhead generation is incremented.

## Pcap to Netflow Phase
First, pcap files are converted to data stream using [Argus](https://openargus.org/) tool. We have considered all the features that Argus is capable of producing.
Second, we consider different scales of slicing in Argus from 0.001 seconds to 60 seconds, this will cause a change in the number of output Netflows.

## Preprocess Phase
In this section, all Netflow files convert to csv file. By following command you can preprocess .flow files to csv files.
The most important operations in the pre-processing:
- Handle missing values
- Drop columns with very little information
- One Hot Encoding for non-numeric columns
- Min-Max Scaling
NOTE: due to use from one hot encoding columns in each output csv is not exactly same.
set project_conf.yaml file for input and output directory for example:

original_dataset_path: "original_dataset\\scale_30\\SFTP\\flow\\all\\"
preprossed_dataset_path: "preprocessed_dataset\\scale_30\\SFTP\\all\\"

Run preprocess code:

    python preprocesses_netflow.py

## Standardization and combination of CSV files
In order to equalize the number of features:

Run following command for integrate preprocessed dataset:
    
    python standard_pre_data_integration.py

Run following command for create 55 features for all class of dataset:

    python dataset_stanard.py

NOTE: standard_dataset is for analysis each class in different of congestion in network.

For combination of different classes at different levels:

Run following command:

    python dataset_merger.py

NOTE: merged_standard_dataset is for analysis multi label classifier in various level on congestion.



