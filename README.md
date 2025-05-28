# Unleashing the Power of Reinforcement Learning for Trading
## Overview
This project implements a reinforcement learning system for stock trading using technical indicators. I've retrained the model, debugged, and modified the original code to create a functional trading agent that learns to make buy/sell decisions based on market conditions.

Key improvements made:
- Fixed data loading issues with Yahoo Finance API
- Debugged environment and agent implementation
- Enhanced visualization of trading performance
- Implemented Monte Carlo testing for robust evaluation
- Added multi-stock training capability
- Optimized neural network architecture

## Features
- Technical Indicators: EMA, MACD, RSI, OBV, Bollinger Bands
- Reinforcement Learning: Deep Q-Network (DQN) agent
- Trading Environment: Custom OpenAI Gym environment
- Visualization: Interactive trading history and portfolio performance
- Robust Testing: Monte Carlo simulation for performance evaluation
- Multi-Stock Training: Model learns from multiple stock datasets

## Requirements
```bash
pip install gym yfinance ta numpy pandas matplotlib seaborn scikit-learn tensorflow
```
## Usage
1. Run the Jupyter notebook Trading_with_Reinforcement_Learning.ipynb
2. The notebook will:
- Download stock data
- Preprocess data and add technical indicators
- Define the trading environment
- Train the DQN agent
- Test the agent and visualize results
- Perform Monte Carlo simulations

## Key Components
1. Data Preparation
- Downloads historical stock data using Yahoo Finance API
- Adds technical indicators:
  - Exponential Moving Averages (EMA 7, 14, 50, 200)
  - Moving Average Convergence Divergence (MACD)
  - Relative Strength Index (RSI)
  - On-Balance Volume (OBV)
  - Bollinger Bands (BB)

2. Trading Environment
- Custom OpenAI Gym environment
- Action space: [action_type, shares] where:
  - action_type: 0 (Buy), 1 (Sell), 2 (Hold)
  - shares: 0-9 (number of shares to trade)
- State space: 12 technical indicators
- Reward function: Change in portfolio value

3. DQN Agent
- Neural network architecture:
  - Input layer: 12 neurons (technical indicators)
  - Hidden layers: 128, 256, 128 neurons with ReLU activation
  - Output layer: 30 neurons (3 actions × 10 share quantities)
- Experience replay with memory buffer
- Epsilon-greedy exploration strategy

4. Training & Testing
- Trained on multiple stocks (GOOG, IBM, AAPL, META, AMZN)
- Monte Carlo testing for performance evaluation
- Portfolio performance visualization:
  - Trading decisions (buy/sell points)
  - Portfolio value change over time

## Results
The agent learns to make trading decisions based on technical indicators. The visualization shows:
1. Price chart with buy/sell markers
2. Portfolio percentage change
3. Average reward from Monte Carlo simulations

## Tips for Improvement
1. Normalize input data:

```python
from sklearn.preprocessing import MinMaxScaler

def _normalize_data(self, data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data.reshape(1, -1))
```

2. Enhance neural network:
```python
model = Sequential()
model.add(Dense(128, input_dim=state_size, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(action_size, activation='linear'))
```

3. Increase memory buffer:

```python
self.memory = deque(maxlen=5000)  # Original was 2000
```

4. Adjust hyperparameters:

```python
self.gamma = 0.99  # Discount factor
self.epsilon_decay = 0.999  # Slower decay
self.learning_rate = 0.001  # Smaller learning rate
```

5. Add more technical indicators:

```python
df['Stochastic_Oscillator'] = ta.momentum.StochasticOscillator(
    high=df['High'], 
    low=df['Low'], 
    close=df['Close'], 
    window=14
).stoch()
```

## Disclaimer
This project is for educational purposes only. The trading strategies implemented are not financial advice. Always conduct thorough research and consider consulting with a qualified financial advisor before making investment decisions.

## Future Enhancements
- Implement Double DQN or Dueling DQN for better stability
- Add risk management features (stop-loss, take-profit)
- Incorporate fundamental analysis data
- Develop ensemble models with different technical indicators
- Create a live trading interface with brokerage API integration

For a detailed walkthrough of the implementation, please see the Jupyter notebook.
