#!/bin/bash

# Loop through numbers 1 to 14 for argus file creation
for i in {1..14}
do
   # Execute the command with the current number
   rm argus/sftp_filter_mirror_${i}.argus
   argus -r pcap/sftp_filter_mirror_${i}.pcapng -w argus/sftp_filter_mirror_${i}.argus
   echo "Executing: argus_${i}"
done

# Flow creation
# Define the list of strings
strings=("all" "size" "time" "freq" "state" "rate")

# Outer loop over each string
for str in "${strings[@]}"
do
   # Inner loop through numbers 1 to 14
   for i in {1..14}
   do
      # Construct the command string and execute it
      rm flow/${str}/sftp_${str}_mirror_${i}.flow
      ra -F /home/argus/ra_conf/ra_${str}_based.conf -r argus/sftp_filter_mirror_${i}.argus -n -Z b > flow/${str}/sftp_${str}_mirror_${i}.flow
      # Print the command for debugging
      echo "Executing: ${str}_${i}" 
   done
done

