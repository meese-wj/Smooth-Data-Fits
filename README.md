# Smooth Numerical or Experimental Data Fitter via Monte Carlo Simulation
### W. Joe Meese

This Python script smooths out noisy one-dimensional data by optimizing a cost function that has two competing parts. The terms are 

1. The square difference between the smooth model and the input data.
1. The square difference between the derivative of the smooth model at each input data point.

The competition is quantified by a stiffness coefficient multiplying the derivative term. For a _stiffer_ cost function, the model will typically pull away from the noisy data and look _smoother_. However, this is typically at the cost (:grin:) of being a _poorer_ fit to the numerical/experimental data.

## Notes on nomenclature
Here I clarify some of the terminology I use in the repo:
* `input`: the arugment (x-axis) of the data set used. This is never modified.
* `data`: the numerical or experimental values (y-axis) in the data set used. This is also never modified.
* `model`: this is the modified set of data. The model is of identical `shape` to the `data`, and its values are iteratively updated to satisfy the cost function.