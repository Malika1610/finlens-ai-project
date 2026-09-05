def fundamental_agent(stock):
    pe = stock["pe"]
    revenue_growth = stock["revenue_growth"]

    # Original decision logic preserved
    if revenue_growth > 10 and pe < 30:
        signal = "BULLISH"
        confidence = 0.80
        reason = "Strong revenue growth combined with a reasonable valuation."

    elif revenue_growth < 0 or pe > 50:
        signal = "BEARISH"
        confidence = 0.75
        reason = "Weak growth or relatively high valuation increases fundamental risk."

    else:
        signal = "NEUTRAL"
        confidence = 0.65
        reason = "Fundamental indicators are mixed."

    return {
        "agent": "Fundamental Analyst",
        "signal": signal,
        "confidence": confidence,
        "reasoning": reason,
        "evidence": {
            "PE_ratio": pe,
            "revenue_growth": revenue_growth
        }
    }