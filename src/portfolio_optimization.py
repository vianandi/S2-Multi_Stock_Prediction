import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.risk_management import calculate_returns, calculate_volatility

def calculate_portfolio_return(weights, returns):
    """Hitung expected return portfolio"""
    return np.sum(returns * weights)

def calculate_portfolio_volatility(weights, cov_matrix):
    """Hitung portfolio volatility"""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def portfolio_performance(weights, mean_returns, cov_matrix):
    """Hitung return dan risk portfolio"""
    returns = calculate_portfolio_return(weights, mean_returns)
    volatility = calculate_portfolio_volatility(weights, cov_matrix)
    return returns, volatility

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.06):
    """
    Negative Sharpe ratio untuk minimization
    """
    p_returns, p_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    if p_volatility == 0:
        return 1000  # Large penalty
    return -(p_returns - risk_free_rate) / p_volatility

def max_sharpe_ratio_optimization(mean_returns, cov_matrix, risk_free_rate=0.06):
    """
    Optimasi portfolio untuk Maximum Sharpe Ratio
    """
    num_assets = len(mean_returns)
    
    # Constraints: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds: 0 <= weight <= 1 (no short selling)
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Initial guess: equal weights
    init_guess = num_assets * [1. / num_assets]
    
    # Optimization
    result = minimize(negative_sharpe, init_guess,
                     args=(mean_returns, cov_matrix, risk_free_rate),
                     method='SLSQP',
                     bounds=bounds,
                     constraints=constraints)
    
    return result.x

def minimum_variance_optimization(mean_returns, cov_matrix):
    """
    Optimasi portfolio untuk Minimum Variance (risk)
    """
    num_assets = len(mean_returns)
    
    def portfolio_variance(weights):
        return calculate_portfolio_volatility(weights, cov_matrix)
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets]
    
    result = minimize(portfolio_variance, init_guess,
                     method='SLSQP',
                     bounds=bounds,
                     constraints=constraints)
    
    return result.x

def efficient_frontier(mean_returns, cov_matrix, num_portfolios=50):
    """
    Generate efficient frontier dengan berbagai target returns
    """
    results = []
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), num_portfolios)
    
    for target in target_returns:
        weights = target_return_optimization(mean_returns, cov_matrix, target)
        ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
        results.append({
            'return': ret,
            'volatility': vol,
            'sharpe': (ret - 0.06) / vol if vol > 0 else 0,
            'weights': weights
        })
    
    return pd.DataFrame(results)

def target_return_optimization(mean_returns, cov_matrix, target_return):
    """
    Optimasi untuk mencapai target return dengan minimum risk
    """
    num_assets = len(mean_returns)
    
    def portfolio_variance(weights):
        return calculate_portfolio_volatility(weights, cov_matrix)
    
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: calculate_portfolio_return(x, mean_returns) - target_return}
    )
    
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets]
    
    result = minimize(portfolio_variance, init_guess,
                     method='SLSQP',
                     bounds=bounds,
                     constraints=constraints)
    
    if result.success:
        return result.x
    else:
        # Fallback to equal weights
        return np.array(init_guess)

def risk_parity_optimization(cov_matrix):
    """
    Risk Parity: alokasi berdasarkan equal risk contribution
    """
    num_assets = len(cov_matrix)
    
    def risk_contribution(weights):
        portfolio_vol = calculate_portfolio_volatility(weights, cov_matrix)
        marginal_contrib = np.dot(cov_matrix, weights)
        risk_contrib = weights * marginal_contrib / portfolio_vol
        
        # Minimize sum of squared differences from equal risk
        target_risk = portfolio_vol / num_assets
        return np.sum((risk_contrib - target_risk) ** 2)
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets]
    
    result = minimize(risk_contribution, init_guess,
                     method='SLSQP',
                     bounds=bounds,
                     constraints=constraints)
    
    return result.x

def monte_carlo_portfolios(mean_returns, cov_matrix, num_portfolios=10000):
    """
    Generate random portfolios untuk visualisasi
    """
    num_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        # Random weights
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        # Calculate metrics
        portfolio_return, portfolio_std = portfolio_performance(weights, mean_returns, cov_matrix)
        
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std
        results[2, i] = (portfolio_return - 0.06) / portfolio_std  # Sharpe
    
    return results, weights_record

def optimize_portfolio(stock_data_dict, method='max_sharpe'):
    """
    Main function untuk portfolio optimization
    
    Parameters:
    -----------
    stock_data_dict : dict
        Dictionary dengan format {stock_code: price_array}
    method : str
        'max_sharpe', 'min_variance', 'risk_parity'
    
    Returns:
    --------
    dict dengan optimal weights dan metrics
    """
    # Prepare data
    stock_codes = list(stock_data_dict.keys())
    returns_data = []
    
    for code in stock_codes:
        prices = stock_data_dict[code]
        returns = calculate_returns(prices)
        returns_data.append(returns)
    
    # Ensure all have same length
    min_len = min(len(r) for r in returns_data)
    returns_data = [r[-min_len:] for r in returns_data]
    
    # Create returns matrix
    returns_matrix = np.array(returns_data).T
    mean_returns = np.mean(returns_matrix, axis=0) * 252  # Annualized
    cov_matrix = np.cov(returns_matrix.T) * 252  # Annualized
    
    # Optimize based on method
    if method == 'max_sharpe':
        optimal_weights = max_sharpe_ratio_optimization(mean_returns, cov_matrix)
    elif method == 'min_variance':
        optimal_weights = minimum_variance_optimization(mean_returns, cov_matrix)
    elif method == 'risk_parity':
        optimal_weights = risk_parity_optimization(cov_matrix)
    else:
        # Default: equal weights
        optimal_weights = np.array([1/len(stock_codes)] * len(stock_codes))
    
    # Calculate portfolio metrics
    port_return, port_volatility = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
    port_sharpe = (port_return - 0.06) / port_volatility if port_volatility > 0 else 0
    
    # Create result
    result = {
        'stocks': stock_codes,
        'weights': optimal_weights,
        'expected_return': port_return,
        'volatility': port_volatility,
        'sharpe_ratio': port_sharpe,
        'method': method
    }
    
    # Add allocation details
    allocation = []
    for code, weight in zip(stock_codes, optimal_weights):
        if weight > 0.01:  # Only show allocations > 1%
            allocation.append({
                'stock': code,
                'weight': weight,
                'weight_pct': weight * 100
            })
    
    result['allocation'] = sorted(allocation, key=lambda x: x['weight'], reverse=True)
    
    return result

def calculate_diversification_ratio(weights, cov_matrix):
    """
    Diversification Ratio = Weighted Average Volatility / Portfolio Volatility
    Higher is better (more diversified)
    """
    individual_vols = np.sqrt(np.diag(cov_matrix))
    weighted_vol = np.sum(weights * individual_vols)
    portfolio_vol = calculate_portfolio_volatility(weights, cov_matrix)
    
    if portfolio_vol == 0:
        return 1.0
    
    return weighted_vol / portfolio_vol
