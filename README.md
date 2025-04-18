<pre>
# Minecraft Skill Data 

This dataset and toolkit build upon the foundation of OpenAI's VPT models, extending them with enhanced planning and visualization tools for Minecraft skill learning. We include tools for **automatic labelling**, **PCA feature extraction**, and **action-feature alignment**. 

## Features

- **Segmented Skill Traces:** Automatically segment and label skill traces using inventory data and environment transitions.
- **Multiple Trace Formats:** Export raw POV pixel observations, as well as PCA features. 
- **PCA Feature Extraction:** Apply PCA to frames for low-dimensional feature extraction.
- **Ground Truth & Mapping:** Store accurate labels, goals, and world state mappings alongside each skill.

## Installation

Install using instructions found on the [VPT Github](https://github.com/openai/Video-Pre-Training?tab=readme-ov-file) 

## Usage

### Generating data 
Use the included script to generate labelled data from the environment. This script can be modified to change the target item for the agent. 
```bash
bash generate_data.sh
```

### Running the PCA features 
Again, using the included python file will generate PCA features from the data above. 
```bash
python pca_features.py
``` 
 

### Downloading the Expert Dataset
Use the script to pull down expert data (generated with VPT and labelled via inventory deltas):

```bash
python download_dataset.py
```



## Folder Structure

```
Data/
    skill_name/
        pca_features/
        pixel_obs/
        groundTruth/
        mapping/
```

