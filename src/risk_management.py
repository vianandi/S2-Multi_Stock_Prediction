import numpy as np
import pandas as pd
from scipy import stats

def calculate_returns(prices):
    """Hitung daily returns dari price series"""
    return np.diff(prices) / prices[:-1]

def calculate_volatility(prices, period=252):
    """
    Hitung annualized volatility (standard deviation of returns)
    period=252 untuk trading days per year
    """
    returns = calculate_returns(prices)
    return np.std(returns) * np.sqrt(period)

def calculate_sharpe_ratio(prices, risk_free_rate=0.06, period=252):
    """
    Sharpe Ratio = (Return - Risk Free Rate) / Volatility
    risk_free_rate default 6% (BI rate Indonesia)
    """
    returns = calculate_returns(prices)
    avg_return = np.mean(returns) * period
    volatility = np.std(returns) * np.sqrt(period)
    
    if volatility == 0:
        return 0
    
    sharpe = (avg_return - risk_free_rate) / volatility
    return sharpe

def calculate_sortino_ratio(prices, risk_free_rate=0.06, period=252):
    """
    Sortino Ratio = (Return - Risk Free Rate) / Downside Deviation
    Hanya menghitung volatility dari negative returns
    """
    returns = calculate_returns(prices)
    avg_return = np.mean(returns) * period
    
    # Downside deviation (hanya negative returns)
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        return float('inf')
    
    downside_dev = np.std(negative_returns) * np.sqrt(period)
    
    if downside_dev == 0:
        return 0
    
    sortino = (avg_return - risk_free_rate) / downside_dev
    return sortino

def calculate_max_drawdown(prices):
    """
    Maximum Drawdown = Maximum peak-to-trough decline
    Mengukur worst-case scenario loss
    """
    cumulative = np.maximum.accumulate(prices)
    drawdown = (prices - cumulative) / cumulative
    max_dd = np.min(drawdown)
    return max_dd

def calculate_var(prices, confidence=0.95):
    """
    Value at Risk (VaR) - Historical method
    confidence=0.95 means 95% confident losses won't exceed VaR
    """
    returns = calculate_returns(prices)
    var = np.percentile(returns, (1 - confidence) * 100)
    return var

def calculate_cvar(prices, confidence=0.95):
    """
    Conditional Value at Risk (CVaR) / Expected Shortfall
    Average loss beyond VaR threshold
    """
    returns = calculate_returns(prices)
    var = calculate_var(prices, confidence)
    cvar = returns[returns <= var].mean()
    return cvar

def calculate_beta(stock_prices, market_prices):
    """
    Beta = Covariance(stock, market) / Variance(market)
    Mengukur systematic risk relatif terhadap market
    """
    stock_returns = calculate_returns(stock_prices)
    market_returns = calculate_returns(market_prices)
    
    # Ensure same length
    min_len = min(len(stock_returns), len(market_returns))
    stock_returns = stock_returns[-min_len:]
    market_returns = market_returns[-min_len:]
    
    covariance = np.cov(stock_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)
    
    if market_variance == 0:
        return 1.0
    
    beta = covariance / market_variance
    return beta

def calculate_all_risk_metrics(prices, market_prices=None):
    """
    Hitung semua risk metrics untuk satu saham
    Returns dictionary dengan semua metrics
    """
    metrics = {
        'volatility': calculate_volatility(prices),
        'sharpe_ratio': calculate_sharpe_ratio(prices),
        'sortino_ratio': calculate_sortino_ratio(prices),
        'max_drawdown': calculate_max_drawdown(prices),
        'var_95': calculate_var(prices, 0.95),
        'cvar_95': calculate_cvar(prices, 0.95),
        'var_99': calculate_var(prices, 0.99),
        'cvar_99': calculate_cvar(prices, 0.99),
    }
    
    if market_prices is not None:
        metrics['beta'] = calculate_beta(prices, market_prices)
    
    return metrics

def position_sizing_kelly(win_rate, avg_win, avg_loss):
    """
    Kelly Criterion untuk optimal position sizing
    f = (p*b - q) / b
    where p=win_rate, q=1-p, b=avg_win/avg_loss
    """
    if avg_loss == 0:
        return 0
    
    b = abs(avg_win / avg_loss)
    q = 1 - win_rate
    kelly = (win_rate * b - q) / b
    
    # Conservative: use half Kelly
    return max(0, min(kelly * 0.5, 1.0))

def position_sizing_risk_based(capital, risk_per_trade, entry_price, stop_loss_price):
    """
    Position sizing based on fixed risk per trade
    risk_per_trade: percentage of capital to risk (e.g., 0.02 for 2%)
    """
    risk_amount = capital * risk_per_trade
    price_risk = abs(entry_price - stop_loss_price)
    
    if price_risk == 0:
        return 0
    
    shares = risk_amount / price_risk
    return int(shares)

def calculate_stop_loss_take_profit(current_price, atr, risk_reward_ratio=2.0):
    """
    Hitung stop loss dan take profit berdasarkan ATR
    risk_reward_ratio: TP distance = SL distance * ratio
    """
    stop_loss = current_price - (2 * atr)  # 2 ATR di bawah current price
    risk = current_price - stop_loss
    take_profit = current_price + (risk * risk_reward_ratio)
    
    return {
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'risk': risk,
        'reward': take_profit - current_price,
        'risk_reward_ratio': risk_reward_ratio
    }

def risk_assessment(metrics):
    """
    Berikan risk assessment berdasarkan metrics
    Returns: 'Low', 'Medium', 'High' risk level
    """
    risk_score = 0
    
    # Volatility check
    if metrics['volatility'] > 0.4:  # >40% annual volatility
        risk_score += 2
    elif metrics['volatility'] > 0.25:  # >25%
        risk_score += 1
    
    # Sharpe ratio check (lower is worse)
    if metrics['sharpe_ratio'] < 0.5:
        risk_score += 2
    elif metrics['sharpe_ratio'] < 1.0:
        risk_score += 1
    
    # Max drawdown check
    if metrics['max_drawdown'] < -0.3:  # >30% drawdown
        risk_score += 2
    elif metrics['max_drawdown'] < -0.2:  # >20%
        risk_score += 1
    
    # VaR check
    if metrics['var_95'] < -0.05:  # >5% daily loss
        risk_score += 1
    
    if risk_score >= 4:
        return 'High'
    elif risk_score >= 2:
        return 'Medium'
    else:
        return 'Low'
