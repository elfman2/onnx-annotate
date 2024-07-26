# onnx-annotate
This repository contains artefacts mentionned in the paper Date 2025 : "ONNX as the support of Airborne Machine Learning Model Description":
* Aerospace Recommended Practice ARP-6983 Machine Learning objectives and standards.
* CNN Training Framework Models (TFM) examples in keras and pytorch 
* their generated Machine Learning Model Description (MLMD) instances
* ONNX annotation tools which were developped and used for the experiments

## Prerequisites

    pip install -r requirements.txt

## Build
To build MLMD instances out of Training Framework Model (TFM)

    inv build

## Browse the doc
Documents are in the form of [strictdoc](https://strictdoc.readthedocs.io/en/stable/)
To browse all documents using strictdoc server on port 8081

    cd doc
    strictdoc server . 

server params are in [strictdoc.toml](doc/strictdoc.toml)

To export to html all documents

    strictdoc export . 

## Clean

    inv clean

## [TFM](TFM) 
contains the Training Framework Models

## [MLMD](MLMD)
contains ARP-6983 ONNX MLMD instances 

## [annotate](annotate) 
contains annotation script for MLMD

## [standards](doc/01_standards)
contains: 
* ARP-6983 related objectives.
* MLMD and TFM standard related to ARP-6983 objectives.

## [specifications](doc/02_specifications)
contains the TFM and MLMD generated requirements and traceability in the form of [strictdoc](https://strictdoc.readthedocs.io/en/stable/)