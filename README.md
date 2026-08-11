# Hybrid Physics-Informed Machine Learning for PEMFC Degradation Modeling

A hybrid physics-informed machine learning framework for long-term
PEM fuel cell voltage prediction and degradation analysis.

## Overview

This project combines an electrochemical voltage model with machine
learning to model PEM fuel cell degradation under dynamic operating
conditions.

The physics-based model describes the main voltage losses, while
machine learning is used to model the remaining residual behavior.

## Physics-Based Model

The cell voltage is represented as:

V_cell = E_OCV - a ln(I) - R_ohmic I - c sqrt(I)

where:

- E_OCV is the open-circuit voltage
- a represents the activation loss factor
- R_ohmic is the ohmic resistance
- c represents the concentration/mass-transport loss factor
- I is the current

## Hybrid Model

The residual is calculated as:

Residual = Measured Voltage - Physics Prediction

The residual is separated into:

Residual = Degradation + Operating-condition effects

Ridge regression is used to model the long-term degradation trend,
while XGBoost is used to model operating-condition-related variations.

## Results

The model is evaluated using a chronological train/test split.

Training data:
0–800 h

Testing data:
800 h+

Physics model only:
RMSE = 59.40 mV, MAE = 52.81 mV, R2 = -1.6317

Physics + Degradation + XGBoost:
RMSE = 19.61 mV, MAE = 15.90 mV, R2 = 0.7133

RMSE reduction = 66.99%

## Limitations

- Polarization and EIS measurements are available only at selected
  ageing times.
- Electrochemical parameters are interpolated between characterization
  snapshots.
- Beyond the final available characterization snapshot, the last
  parameter values are held constant.
- Future operating conditions are assumed to be known.
- Validation is performed using a chronological train/test split.
- Further validation using independent PEMFC datasets is required.
