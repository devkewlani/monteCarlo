import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt  # Import for plotting
matplotlib.use('Agg')  # Use Agg backend for non-interactive environments

class OptionPricing:

    def __init__(self, currPrice, expPrice, rf, sigma, T, iters, steps):
        self.S = currPrice
        self.E = expPrice
        self.rf = rf
        self.sigma = sigma
        self.T = T
        self.iters = iters
        self.steps = steps  # Number of time steps for each path

    def generate_price_paths(self):

        # Time increment
        dt = self.T / self.steps

        # Initialize the stock price paths matrix (iters x steps)
        price_paths = np.zeros((self.iters, self.steps + 1))
        price_paths[:, 0] = self.S  # All paths start with the current price

        # Generate random Brownian motion increments
        for t in range(1, self.steps + 1):
            # Random normal values for Brownian motion
            z = np.random.normal(0, 1, self.iters)
            # Apply the GBM formula for each time step
            price_paths[:, t] = price_paths[:, t-1] * np.exp(
                (self.rf - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z
            )

        return price_paths

    def call_option_price(self, price_paths):
        # Payoff for call option: max(S_T - E, 0)
        payoff = np.maximum(price_paths[:, -1] - self.E, 0)
        # Discount factor
        discount_factor = np.exp(-self.rf * self.T)
        # Average discounted payoff
        return discount_factor * np.mean(payoff)

    def put_option_price(self, price_paths):
        # Payoff for put option: max(E - S_T, 0)
        payoff = np.maximum(self.E - price_paths[:, -1], 0)
        # Discount factor
        discount_factor = np.exp(-self.rf * self.T)
        # Average discounted payoff
        return discount_factor * np.mean(payoff)

if __name__ == "__main__":

    currPrice = 100
    strikePrice = 110
    expiry = 1
    rfr = 0.05
    sigma = 0.2
    iters = 10000
    steps = 100  # Number of time steps in each simulation path

    model = OptionPricing(currPrice, strikePrice, rfr, sigma, expiry, iters, steps)
    
    # Generate stock price paths
    price_paths = model.generate_price_paths()

    # Calculate call and put option prices
    call_price = model.call_option_price(price_paths)
    put_price = model.put_option_price(price_paths)

    # Plot the simulated stock price paths
    plt.figure(figsize=(10, 6))
    plt.plot(price_paths.T, lw=1, alpha=0.6)  # Transpose to plot each path
    plt.title(f"Simulated Stock Price Paths - {iters} Simulations, {steps} Steps")
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price")
    plt.grid(True)

    # Save the plot to a file
    plt.savefig('stock_price_paths.png')

    # Output option prices
    print(f"Call option price: {call_price}")
    print(f"Put option price: {put_price}")
