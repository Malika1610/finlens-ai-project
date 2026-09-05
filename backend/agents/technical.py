def technical_agent(stock):
    price = stock["price"]
    change = stock["change"]
    volume = stock["volume"]

    # Original decision logic preserved
    if change >= 2:
        signal = "BULLISH"
        confidence = 0.82
        reason = f"Price momentum is positive at +{change}%."
    elif change <= -2:
        signal = "BEARISH"
        confidence = 0.82
        reason = f"Price momentum is negative at {change}%."
    else:
        signal = "NEUTRAL"
        confidence = 0.65
        reason = "Price movement is relatively stable."

    return {
        "agent": "Technical Analyst",
        "signal": signal,
        "confidence": confidence,
        "reasoning": reason,
        "evidence": {
            "price": price,
            "daily_change": change,
            "volume": volume
        }
    }