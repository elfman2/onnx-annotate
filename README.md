# onnx-annotate
This repository contains artefacts mentionned in the paper Date 2025 : "ONNX as the support of Airborne Machine Learning Model Description":
* Aerospace Recommended Practice ARP-6983 Machine Learning objectives and standards.
* CNN model examples
* their Machine Learning Model Description (MLMD) instances
* ONNX annotation tools which were developped and used for the experiments

To build MLMD instances out of Training Framework Model (TFM)

    pip install -r requirements.txt
    inv build

## [TFM](TFM) 
contains the Training Framework Models

## [MLMD](MLMD)
contains ARP-6983 MLMD instances 

## [annotate](annotate) 
contains annotation script for MLMD

## [standards](standards)
contains: 
* ARP-6983 related objectives.
* MLMD and TFM standard related to ARP-6983 objectives.
