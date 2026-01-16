def make_decision(current, forecast, threshold=0.02):
    change = (forecast-current)/current
    if change > threshold:
        return "BUY", change*100
    elif change < -threshold:
        return "SELL", change*100
    else:
        return "HOLD", change*100