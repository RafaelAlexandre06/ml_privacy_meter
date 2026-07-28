#!/usr/bin/env bash
python3 run_urmia.py --cf configs/urmia/cifar10_neggrad_plus.yaml
python3 run_urmia_online.py --cf configs/urmia/cifar10_online.yaml
python3 run_urmia_online_wig.py --cf configs/urmia/cifar10_online_wig.yaml
