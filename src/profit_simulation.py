def simulate(dss, capital=100_000_000):
    invest = capital/len(dss)
    profit=0

    for row in dss:
        if row["Decision"]=="BUY":
            shares = invest/row["Current"]
            profit += (row["Forecast"]-row["Current"])*shares

    return profit, profit/capital*100