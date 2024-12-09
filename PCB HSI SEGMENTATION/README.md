# Optimizing Industrial E-Waste Recycling with Attention-Driven Deep Learning for PCB Segmentation Using Hyperspectral Imaging

## Overview

The aim of this research work is to enhance the efficacy of E-waste recycling, specifically printed circuit boards (PCBs). By exploiting hyperspectral imaging (HSI), which offers spectroscopic analysis to accurately identify materials, this paper presents attention-based deep learning segmentation models to accurately identify components in PCBs. This approach allows for the automatic extraction of information from E-waste, leading to more efficient and optimized recycling practices.

## Dataset Details

The dataset includes:
- RGB images of 53 PCBs scanned with a high-resolution RGB camera (Teledyne Dalsa C4020).
- 53 hyperspectral data cubes of those PCBs scanned with Specim FX10 in the VNIR range.
- Two segmentation ground truth files: 'General' and 'Monoseg' for 4 classes of interest - 'others,' 'IC,' 'Capacitor,' and 'Connectors.'
*  In this experiment, hyperspectral data cubes and 'General' ground truth are used.
![http://url/to/img.png](https://github.com/Elias-Arbash/PCBVision/blob/main/images/training_hsi.png)

## Acknowledgments

The main structure of this project was adapted from [https://github.com/hifexplo/PCBVision] (link to the original source). 
I would like to thank the authors for their work, which served as the foundation for this project.


## Requirements

Before running the code, make sure to install all the dependencies specified in the Requirements.txt file. Note that a GPU is required for processing and handling the data effectively.

## Data Access

To utilize the dataset, download it from [this link](https://rodare.hzdr.de/record/2704).


