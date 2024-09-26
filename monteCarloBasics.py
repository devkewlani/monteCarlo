import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt  # Import for plotting
matplotlib.use('Agg')  # Use Agg backend for non-interactive environments

class OptionPricing:

    def __init__(self, currPrice, expPrice, rf, sigma, T, iters):
        self.S = currPrice
        self.E = expPrice
        self.rf = rf
        self.sigma = sigma
        self.T = T
        self.iters = iters

    def call_options_simulations(self):

        options_data = np.zeros([self.iters, 2])

        bm1 = np.random.normal(0,1, [self.iters, 1])  # Adjust to be a column vector

        stock_price = self.S*np.exp(((self.rf - 0.5*(self.sigma**2))*self.T) + self.sigma*np.sqrt(self.T)*bm1)

        options_data[:,1] = stock_price[:, 0] - self.E  # Calculate the payoff
        average = np.sum(np.amax(options_data, axis=1))/float(self.iters)

        discount_factor = np.exp(-self.rf*self.T)

        return discount_factor*average, stock_price
    
    def put_options_simulations(self):

        options_data = np.zeros([self.iters, 2])

        bm1 = np.random.normal(0,1, [self.iters, 1])  # Adjust to be a column vector

        stock_price = self.S*np.exp(((self.rf - 0.5*(self.sigma**2))*self.T) + self.sigma*np.sqrt(self.T)*bm1)

        options_data[:,1] = self.E - stock_price[:, 0]
        average = np.sum(np.amax(options_data, axis=1))/float(self.iters)

        discount_factor = np.exp(-self.rf*self.T)

        return discount_factor*average, stock_price
    
if __name__ == "__main__":

    currPrice = 100
    strikePrice = 110
    expiry = 1
    rfr = 0.05
    sigma = 0.1
    iters = 10000

    model = OptionPricing(currPrice, strikePrice, rfr, sigma, expiry, iters)
    
    # Simulate call and put options
    call_price, call_paths = model.call_options_simulations()
    put_price, put_paths = model.put_options_simulations()

    # Plot the stock price paths for call option simulations
    plt.figure(figsize=(10, 6))
    plt.plot(call_paths, color='blue', alpha=0.5)
    plt.title(f"Simulated Stock Price Paths - Call Option Pricing ({iters} iterations)")
    plt.xlabel('Simulation Index')
    plt.ylabel('Stock Price')
    plt.grid(True)
    # plt.show()

    # Save the plot as a file (as Agg is non-interactive)
    plt.savefig('call_option_paths.png')
    print(f"Call option price: {call_price}")
    print(f"Put option price: {put_price}")
