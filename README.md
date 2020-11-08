# sim-netdata

The goal of this repo is to simulate Gaussian data with a covariance structure
encoded by a (currently, unweighted and undirected) network. This repo is still
in-progress and has not been tested comprehensively yet.

## Conceptual goals

Given an input network, we want:
1. If variables are connected in the input network, they should be correlated
   in the sample data.
2. Variables with the same sign in the input network should be correlated in
   the same direction once data is sampled.

TODO: write up explanation of why setting correlation matrix directly doesn't
work, make some visualizations, etc.
