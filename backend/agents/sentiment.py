def sentiment_agent(stock):
    sentiment = stock["sentiment"]

    # Original decision logic preserved
    if sentiment > 0.5:
        signal = "BULLISH"
        confidence = 0.78
        reason = "Recent market/news sentiment is positive."

    elif sentiment < -0.5:
        signal = "BEARISH"
        confidence = 0.78
        reason = "Recent market/news sentiment is negative."

    else:
        signal = "NEUTRAL"
        confidence = 0.60
        reason = "Market sentiment is mixed."

    return {
        "agent": "Sentiment Analyst",
        "signal": signal,
        "confidence": confidence,
        "reasoning": reason,
        "evidence": {
            "sentiment_score": sentiment
        }
    }