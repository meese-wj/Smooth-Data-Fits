import numpy as np
import matplotlib.pyplot as plt 

"""
This data fitting protocol essentially uses a Monte
Carlo scheme that minimizes the following competing 
terms:

    Cost = ( model - data )^2 
           + stiffness * ( model_slope_difference )^2

Of course, these terms are summed over all data. The
'stiffness' quantity is a parameter that controls 
the degree to which either term dominates the mini-
mization scheme.

Currently, this data only works for functions of 
1D arguments.
"""

class SmoothDataFitter:
    def __init__(self) -> None:
        pass

    def set_stiffness(self, _stiff):
        self.stiffness = _stiff
        return 
    
    def set_data(self, _data):
        self.data = _data
        return 
    
    def set_input(self, _input):
        self.input = _input
        return 

    def set_problem(self, _input, _data):
        self.set_input(_input)
        self.set_data(_data)
        return
    
    def initialize_model(self):
        self.model = np.zeros( self.data.shape )
        self.model = self.get_average_array(use_data=True)
        return

    def initialize_problem(self, _input, _data, stiffness=1.):
        self.set_stiffness(stiffness)
        self.set_problem(_input, _data)
        self.initialize_model()
        return

    def average_single_point(self, index, data_to_avg):
        dx_left  = self.input[index] - self.input[index - 1]
        dx_right = self.input[index + 1] - self.input[index]
        weights = 1/abs(dx_left), 1/abs(dx_right)
        avg = ( weights[0] * data_to_avg[index - 1] + weights[1] * data_to_avg[index + 1] ) / ( weights[0] + weights[1] )
        return avg
    
    def average_data_interior(self, use_data=True):
        data_to_avg = self.model 
        if use_data:
            data_to_avg = self.data
        averaged_data = np.zeros( data_to_avg.shape )
        for idx in range(1, len(data_to_avg)-1):
            averaged_data[idx] = self.average_single_point( idx, data_to_avg )
        return averaged_data

    def extrapolate_boundary_points(self, averaged_data):
        # This uses a linear extrapolation off of interior
        # to find the boundary. This essentially means that 
        # the smooth fit will be linear at either boundary.
        left_boundary  = averaged_data[1] + ( averaged_data[2] - averaged_data[1] ) / ( self.input[2] - self.input[1] ) * ( self.input[0] - self.input[1] )
        right_boundary = averaged_data[-2] + ( averaged_data[-2] - averaged_data[-3] ) / ( self.input[-2] - self.input[-3] ) * ( self.input[-1] - self.input[-2] )
        return left_boundary, right_boundary

    def get_average_array(self, use_data=True):
        averaged_data = self.average_data_interior(use_data)
        boundary_points = self.extrapolate_boundary_points(averaged_data)
        averaged_data[0] = boundary_points[0]
        averaged_data[-1] = boundary_points[1]
        return averaged_data

    def point_model_diff_term(self, index, step=0.):
        # This term wants to minimize the difference
        # between the model (+step) and the input data
        return ( self.model[index] + step - self.data[index] ) ** 2.
    
    def point_model_slope_term(self, index, step=0.):
        # This term wants to minimize the discontinuity
        # in the derivatives.
        # TODO: figure out a better weighting procedure
        #       for the difference in derivatives
        model_point = self.model[index] + step
        # weights = 1/abs( self.input[index] - self.input[index - 1] ), 1/abs( self.input[index + 1] - self.input[index] )
        weights = 1, 1
        left_deriv  = (model_point - self.model[index - 1]) / (self.input[index] - self.input[index - 1])
        right_deriv = (self.model[index + 1] - model_point) / (self.input[index + 1] - self.input[index])
        return self.stiffness * (( weights[0] * left_deriv - weights[1] * right_deriv ) / ( weights[0] + weights[1] )) ** 2.

    def boundary_slope_term(self, index, step=0.):
        # By definition, the boundary slopes want
        # to be the same as the adjacent interior
        # slopes. 
        # TODO: Make the weights the same as the 
        #       interior
        interior_slope = None
        boundary_slope = None
        model_point = self.model[index] + step
        if index == 0:
            interior_slope = ( self.model[index + 2] - self.model[index + 1] ) / (self.input[index + 2] - self.input[index + 1])
            boundary_slope = (self.model[index + 1] - model_point) / ( self.input[index + 1] - self.input[index] )
        else:
            interior_slope = ( model_point - self.model[index - 1] ) / (self.input[index] - self.input[index - 1])
            boundary_slope = (self.model[index - 1] - self.model[index - 2]) / ( self.input[index - 1] - self.input[index - 2] )

        return self.stiffness * ( interior_slope - boundary_slope ) ** 2.

    def point_cost(self, index, step=0.):
        slope_functions = self.point_model_slope_term, self.boundary_slope_term
        slope_function_idx = int( index == 0 or index == len(self.model)-1 )
        return self.point_model_diff_term(index, step) + slope_functions[slope_function_idx](index, step)

    def model_cost(self):
        # Only compute the cost from the interior
        cost = 0.
        for idx in range(0, len(self.model)):
            cost += self.point_cost(idx)
        return cost

    def model_rmse(self):
        return np.std( self.model - self.data )

    def wiggle(self, tolerance=1e-8, maxiter=int(1e4), check_convergence=50):
        starting_cost = self.model_cost()
        print("Starting cost = ", starting_cost)
        current_cost = starting_cost
        self.costs = np.zeros(maxiter)
        self.costs[0] = current_cost
        iteration = 0
        not_finished = True
        cost_fluctuations = 2**31 - 1
        while not_finished:
            iteration += 1
            max_step_width = self.model_rmse()
            for idx in range(0, len(self.input)):
                step = max_step_width * (-1 + 2 * np.random.rand())
                cost_change = self.point_cost(idx, step=step) - self.point_cost(idx, step=0.)
                self.model[idx] += step * (cost_change < 0.)
            
            previous_cost = current_cost
            current_cost = self.model_cost()
            self.costs[iteration] = current_cost

            if iteration % check_convergence == 0:
                cost_fluctuations = np.var( self.costs[ iteration-check_convergence : iteration ] ) / np.mean( self.costs[ iteration-check_convergence : iteration ] )**2

            finished = (iteration >= maxiter) or (cost_fluctuations < tolerance)
            not_finished = not(finished)

        print("Final cost =", current_cost)
        print("Relative change = {}%".format((current_cost - starting_cost) / starting_cost * 100) )
        print("Cost Fluctuations = {:.3e}".format(cost_fluctuations), "over last", check_convergence, "iterations.")
        print("Total iterations =", iteration)
        return


if __name__ == "__main__":

    test_input = np.array([ 0, 1, 1.5, 2, 2.25, 2.5, 4, 5, 6, 7, 10, 11, 14, 16, 18 ])
    test_data = np.array([ 0, 0.5, 2, 4, 4.1, 4.5, 5, 5.5, 5.25, 4, 2, -2, -2, -1, -0.5 ])


    sdf = SmoothDataFitter()
    sdf.initialize_problem(test_input, test_data, stiffness=1.5)
    weighted_avg = np.copy(sdf.model)
    sdf.wiggle()

    plt.plot(test_input, test_data, "o-", mec="k", label="Test Data")
    # plt.plot(test_input, weighted_avg, "*-", mec="k", label="Weighted Average")
    plt.plot(test_input, sdf.model, "s-", mec="k", label="Monte Carlo Smooth")

    plt.legend()
    plt.show()