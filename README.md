# Explainable Visual Anomaly Detection via Concept Bottleneck Models
This repository contains the code for our work "Explainable Visual Anomaly Detection via Concept Bottleneck Models", through which we try to enhance interpretability in Industrial Visual Anomaly Detection (VAD) by employing Concept Bottleneck Models (CBMs), i.e. neural models that learn meaningful concepts and thus can provide human-understandable descriptions of anomalies and defects. Together with training and testing the standard CBM architectures, we propose a pipeline for generating synthetic anomalies that enables effective concept learning without requiring anomalous samples, thus maintaining traditional VAD requirements. Finally, we propose a modification to the classic CBM architecture to produce not only concept-based explanations but also visual explanations, similar to those in standard VAD models, by integrating a novelty detection branch that leverages features-based VAD algorithms. The experiments were performed starting from the MVTec-AD dataset, which was modified to integrate generated images and concept annotations.

## How to use
The repository is organized as follows:
- [/datasets](/datasets) defines the classes for the original MVTec dataset and the final concept dataset that was used for training the CBMs;
- [/models](/models) contains the scripts necessary to build the models;
- [/trainers](/trainers) and [/evaluators](/evaluators) contain the scripts for respectively training and evaluating both the CBM models and the novelty detection branch for the generation of a heatmap;
- [/main_scripts](/main_scripts) contains the scripts for performing data annotation with concepts, CBM training/testing, novelty detection training/testing, concept evaluation and intervention over concepts.

## Dataset
We tested our dataset creation pipeline starting from the [MVTec-AD dataset](/https://www.mvtec.com/company/research/datasets/mvtec-ad), which was modified to include artificially generated anomalous images, together with the original ones, and concept annotation. To replicate our pipeline for concept extraction and annotation, the MVTec-AD dataset should be stored according to the following structure:
```bash
├── mvtec
│   ├── bottle
│   ├── cable
│   │   ├── ground_truth
│   │   ├── test
│   │   │   ├── bent_wire
│   │   │   │    ├── 000.png
│   │   │   ├── generated anomalies
│   │   │       ├── bent_wire
│   │   │           ├── 000.png
│   │   ├── train
│   │   │   ├── good
│   │   │       ├── 000.png
│   ├── capsule
│   ├── ...
```
