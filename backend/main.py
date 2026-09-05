import os
import time
import json
import math
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf

# ============================================================
# FINLENS AI - REAL COMPANY INTELLIGENCE ENGINE
# ============================================================

app = FastAPI(
    title="FinLens AI Engine",
    description="Multi-agent financial intelligence engine",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# IMPORT AGENTS
# ============================================================

try:
    from backend.agents.technical import technical_agent
    from backend.agents.fundamental import fundamental_agent
    from backend.agents.sentiment import sentiment_agent
except ImportError:
    from agents.technical import technical_agent
    from agents.fundamental import fundamental_agent
    from agents.sentiment import sentiment_agent


# ============================================================
# CONFIGURATION
# ============================================================

SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "FinLensAI research contact@example.com"
)

SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value, default=None):
    """
    Convert a value into a float safely.
    """
    try:
        if value is None:
            return default

        number = float(value)

        if math.isnan(number) or math.isinf(number):
            return default

        return number
    except (ValueError, TypeError):
        return default


def clean_number(value, decimals=2):
    """
    Return a rounded number or None.
    """
    number = safe_float(value)

    if number is None:
        return None

    return round(number, decimals)


def format_large_number(value):
    """
    Convert large numbers into readable values.
    """
    number = safe_float(value)

    if number is None:
        return None

    absolute = abs(number)

    if absolute >= 1_000_000_000_000:
        return f"${number / 1_000_000_000_000:.2f}T"

    if absolute >= 1_000_000_000:
        return f"${number / 1_000_000_000:.2f}B"

    if absolute >= 1_000_000:
        return f"${number / 1_000_000:.2f}M"

    return f"${number:,.0f}"


def calculate_rsi(prices, period=14):
    """
    Calculate RSI using closing prices.
    """
    if len(prices) < period + 1:
        return 50.0

    gains = []
    losses = []

    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]

        if change >= 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    recent_gains = gains[-period:]
    recent_losses = losses[-period:]

    avg_gain = sum(recent_gains) / period
    avg_loss = sum(recent_losses) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss

    return round(100 - (100 / (1 + rs)), 2)


def calculate_max_drawdown(prices):
    """
    Calculate maximum historical drawdown.
    """
    if not prices:
        return 0.0

    peak = prices[0]
    max_drawdown = 0.0

    for price in prices:
        if price > peak:
            peak = price

        if peak > 0:
            drawdown = ((price - peak) / peak) * 100

            if drawdown < max_drawdown:
                max_drawdown = drawdown

    return round(max_drawdown, 2)


def calculate_volatility(prices):
    """
    Calculate annualized realized volatility.
    """
    if len(prices) < 3:
        return 0.0

    returns = []

    for i in range(1, len(prices)):
        previous = prices[i - 1]

        if previous == 0:
            continue

        daily_return = (prices[i] - previous) / previous
        returns.append(daily_return)

    if len(returns) < 2:
        return 0.0

    mean_return = sum(returns) / len(returns)

    variance = sum(
        (r - mean_return) ** 2
        for r in returns
    ) / (len(returns) - 1)

    standard_deviation = math.sqrt(variance)

    annualized = standard_deviation * math.sqrt(252) * 100

    return round(annualized, 2)


def calculate_sma_series(prices, period=20):
    """
    Calculate rolling SMA values.
    """
    result = []

    for i in range(len(prices)):
        start = max(0, i - period + 1)
        window = prices[start:i + 1]

        result.append(
            round(sum(window) / len(window), 2)
        )

    return result


# ============================================================
# SEC HELPERS
# ============================================================

_sec_ticker_cache = None


def sec_request(url):
    """
    Make a request to SEC EDGAR.
    """
    request = urllib.request.Request(
        url,
        headers=SEC_HEADERS
    )

    with urllib.request.urlopen(
        request,
        timeout=8
    ) as response:

        raw = response.read()

        return json.loads(
            raw.decode("utf-8")
        )


def get_sec_ticker_map():
    """
    Download SEC's ticker -> CIK mapping.
    """
    global _sec_ticker_cache

    if _sec_ticker_cache is not None:
        return _sec_ticker_cache

    try:
        url = "https://www.sec.gov/files/company_tickers.json"

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": SEC_USER_AGENT,
                "Accept-Encoding": "gzip, deflate"
            }
        )

        with urllib.request.urlopen(
            request,
            timeout=8
        ) as response:

            data = json.loads(
                response.read().decode("utf-8")
            )

        mapping = {}

        for item in data.values():

            ticker = str(
                item.get("ticker", "")
            ).upper()

            cik = item.get("cik_str")

            if ticker and cik:
                mapping[ticker] = str(cik).zfill(10)

        _sec_ticker_cache = mapping

        return mapping

    except Exception:
        return {}


def get_sec_company_data(ticker):
    """
    Retrieve recent SEC filing information.
    """
    try:
        ticker_map = get_sec_ticker_map()

        cik = ticker_map.get(
            ticker.upper()
        )

        if not cik:
            return {
                "available": False,
                "reason": "SEC CIK not found for this ticker."
            }

        url = (
            f"https://data.sec.gov/submissions/"
            f"CIK{cik}.json"
        )

        data = sec_request(url)

        recent = data.get(
            "filings",
            {}
        ).get(
            "recent",
            {}
        )

        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        accession_numbers = recent.get("accessionNumber", [])
        primary_documents = recent.get(
            "primaryDocument",
            []
        )

        latest = None

        for i, form in enumerate(forms):

            if form in [
                "10-K",
                "10-Q",
                "8-K",
                "20-F",
                "6-K"
            ]:

                filing_date = (
                    filing_dates[i]
                    if i < len(filing_dates)
                    else None
                )

                accession = (
                    accession_numbers[i]
                    if i < len(accession_numbers)
                    else None
                )

                primary_document = (
                    primary_documents[i]
                    if i < len(primary_documents)
                    else None
                )

                latest = {
                    "form": form,
                    "filing_date": filing_date,
                    "accession_number": accession,
                    "primary_document": primary_document
                }

                break

        company_name = data.get(
            "name",
            ticker
        )

        exchange = None

        exchanges = data.get(
            "exchanges",
            []
        )

        if exchanges:
            exchange = exchanges[0]

        return {
            "available": True,
            "cik": cik,
            "company_name": company_name,
            "exchange": exchange,
            "latest_filing": latest,
            "source": "SEC EDGAR"
        }

    except Exception as exc:

        return {
            "available": False,
            "reason": str(exc),
            "source": "SEC EDGAR"
        }


# ============================================================
# COMPANY INFORMATION
# ============================================================

def get_company_information(ticker_obj, ticker):
    """
    Retrieve real company information from Yahoo Finance.
    """

    try:
        info = ticker_obj.get_info()
    except Exception:
        try:
            info = ticker_obj.info
        except Exception:
            info = {}

    if not isinstance(info, dict):
        info = {}

    return {
        "name": info.get(
            "longName"
        ) or info.get(
            "shortName"
        ) or ticker,

        "symbol": ticker,

        "exchange": info.get(
            "exchange"
        ),

        "currency": info.get(
            "currency"
        ),

        "sector": info.get(
            "sector"
        ),

        "industry": info.get(
            "industry"
        ),

        "country": info.get(
            "country"
        ),

        "website": info.get(
            "website"
        ),

        "description": info.get(
            "longBusinessSummary"
        ),

        "employees": info.get(
            "fullTimeEmployees"
        ),

        "market_cap": clean_number(
            info.get("marketCap"),
            0
        ),

        "market_cap_display": format_large_number(
            info.get("marketCap")
        ),

        "enterprise_value": clean_number(
            info.get("enterpriseValue"),
            0
        ),

        "pe_ratio": clean_number(
            info.get("trailingPE")
        ),

        "forward_pe": clean_number(
            info.get("forwardPE")
        ),

        "peg_ratio": clean_number(
            info.get("pegRatio")
        ),

        "price_to_book": clean_number(
            info.get("priceToBook")
        ),

        "profit_margin": clean_number(
            safe_float(info.get("profitMargins")) * 100
            if safe_float(info.get("profitMargins")) is not None
            else None
        ),

        "operating_margin": clean_number(
            safe_float(info.get("operatingMargins")) * 100
            if safe_float(info.get("operatingMargins")) is not None
            else None
        ),

        "return_on_equity": clean_number(
            safe_float(info.get("returnOnEquity")) * 100
            if safe_float(info.get("returnOnEquity")) is not None
            else None
        ),

        "revenue_growth": clean_number(
            safe_float(info.get("revenueGrowth")) * 100
            if safe_float(info.get("revenueGrowth")) is not None
            else None
        ),

        "earnings_growth": clean_number(
            safe_float(info.get("earningsGrowth")) * 100
            if safe_float(info.get("earningsGrowth")) is not None
            else None
        ),

        "dividend_yield": clean_number(
            safe_float(info.get("dividendYield")) * 100
            if safe_float(info.get("dividendYield")) is not None
            else None
        ),

        "beta": clean_number(
            info.get("beta")
        ),

        "52_week_high": clean_number(
            info.get("fiftyTwoWeekHigh")
        ),

        "52_week_low": clean_number(
            info.get("fiftyTwoWeekLow")
        ),

        "target_mean_price": clean_number(
            info.get("targetMeanPrice")
        ),

        "target_high_price": clean_number(
            info.get("targetHighPrice")
        ),

        "target_low_price": clean_number(
            info.get("targetLowPrice")
        )
    }


# ============================================================
# NEWS / SENTIMENT
# ============================================================

def get_news_sentiment(ticker_obj):
    """
    Retrieve recent company news and calculate a simple
    transparent keyword-based sentiment score.

    This is NOT an LLM.
    """

    positive_words = [
        "beat",
        "growth",
        "strong",
        "surge",
        "record",
        "profit",
        "upgrade",
        "bullish",
        "revenue",
        "demand",
        "partnership",
        "innovation",
        "positive",
        "outperform",
        "raises",
        "rises"
    ]

    negative_words = [
        "miss",
        "decline",
        "weak",
        "loss",
        "downgrade",
        "bearish",
        "lawsuit",
        "risk",
        "fall",
        "falls",
        "cut",
        "warning",
        "investigation",
        "concern",
        "layoff",
        "slump"
    ]

    try:
        news = ticker_obj.news
    except Exception:
        try:
            news = ticker_obj.get_news()
        except Exception:
            news = []

    if not isinstance(news, list):
        news = []

    articles = []

    positive_count = 0
    negative_count = 0

    for item in news[:10]:

        if not isinstance(item, dict):
            continue

        content = item.get(
            "content",
            item
        )

        if not isinstance(content, dict):
            content = {}

        title = (
            content.get("title")
            or item.get("title")
            or ""
        )

        publisher = (
            content.get("provider", {})
            if isinstance(content.get("provider"), dict)
            else {}
        )

        publisher_name = (
            publisher.get("displayName")
            or item.get("publisher")
            or "Financial News"
        )

        title_lower = title.lower()

        positive_hits = sum(
            1
            for word in positive_words
            if word in title_lower
        )

        negative_hits = sum(
            1
            for word in negative_words
            if word in title_lower
        )

        positive_count += positive_hits
        negative_count += negative_hits

        link = None

        canonical = content.get(
            "canonicalUrl"
        )

        if isinstance(canonical, dict):
            link = canonical.get(
                "url"
            )

        if not link:
            link = item.get(
                "link"
            )

        articles.append({
            "title": title,
            "publisher": publisher_name,
            "url": link
        })

    total_hits = (
        positive_count +
        negative_count
    )

    if total_hits == 0:
        sentiment_score = 50
        sentiment_label = "NEUTRAL"
    else:
        raw_score = (
            50
            + (
                positive_count -
                negative_count
            ) * 8
        )

        sentiment_score = max(
            0,
            min(100, raw_score)
        )

        if sentiment_score >= 65:
            sentiment_label = "BULLISH"
        elif sentiment_score <= 35:
            sentiment_label = "BEARISH"
        else:
            sentiment_label = "NEUTRAL"

    return {
        "score": sentiment_score,
        "label": sentiment_label,
        "articles": articles,
        "positive_signals": positive_count,
        "negative_signals": negative_count,
        "method": "Transparent keyword sentiment heuristic",
        "source": "Yahoo Finance news feed"
    }


# ============================================================
# RISK ENGINE
# ============================================================

def build_risk_engine(
    price,
    rsi,
    volatility,
    max_drawdown,
    profile,
    capital
):
    """
    Build a transparent risk model.
    """

    risk_points = 0

    # RSI risk
    if rsi >= 70:
        risk_points += 2
    elif rsi >= 60:
        risk_points += 1
    elif rsi <= 30:
        risk_points += 1

    # Volatility risk
    if volatility >= 45:
        risk_points += 3
    elif volatility >= 30:
        risk_points += 2
    elif volatility >= 20:
        risk_points += 1

    # Drawdown risk
    if max_drawdown <= -30:
        risk_points += 3
    elif max_drawdown <= -20:
        risk_points += 2
    elif max_drawdown <= -10:
        risk_points += 1

    if risk_points >= 6:
        risk_level = "HIGH"
    elif risk_points >= 3:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    profile_limits = {
        "Conservative": 0.15,
        "Moderate": 0.35,
        "Aggressive": 0.60
    }

    allocation_percent = profile_limits.get(
        profile,
        0.35
    )

    # Reduce allocation for high risk
    if risk_level == "HIGH":
        allocation_percent *= 0.50
    elif risk_level == "MODERATE":
        allocation_percent *= 0.80

    allocation_percent = round(
        allocation_percent * 100,
        2
    )

    position_value = (
        capital *
        allocation_percent /
        100
    )

    shares = (
        position_value / price
        if price > 0
        else 0
    )

    stop_loss_percent = {
        "Conservative": 3.0,
        "Moderate": 5.0,
        "Aggressive": 7.0
    }.get(
        profile,
        5.0
    )

    stop_loss_price = (
        price *
        (1 - stop_loss_percent / 100)
    )

    take_profit_percent = {
        "Conservative": 5.0,
        "Moderate": 8.0,
        "Aggressive": 15.0
    }.get(
        profile,
        8.0
    )

    take_profit_price = (
        price *
        (1 + take_profit_percent / 100)
    )

    return {
        "risk_level": risk_level,
        "risk_points": risk_points,
        "recommended_allocation_percent": allocation_percent,
        "recommended_position_value": round(
            position_value,
            2
        ),
        "estimated_shares": round(
            shares,
            2
        ),
        "stop_loss_percent": stop_loss_percent,
        "stop_loss_price": round(
            stop_loss_price,
            2
        ),
        "take_profit_percent": take_profit_percent,
        "take_profit_price": round(
            take_profit_price,
            2
        )
    }


# ============================================================
# MARKET WEATHER
# ============================================================

def market_weather(
    trend,
    rsi,
    volatility,
    sentiment_score
):
    """
    Convert quantitative signals into an intuitive
    market weather label.
    """

    bullish_points = 0
    bearish_points = 0

    if trend == "Bullish":
        bullish_points += 2
    else:
        bearish_points += 2

    if rsi >= 55:
        bullish_points += 1
    elif rsi <= 45:
        bearish_points += 1

    if sentiment_score >= 60:
        bullish_points += 1
    elif sentiment_score <= 40:
        bearish_points += 1

    if volatility >= 40:
        weather = "STORMY"
        description = "High volatility environment."
    elif bullish_points >= bearish_points + 2:
        weather = "SUNNY"
        description = "Momentum and sentiment are supportive."
    elif bearish_points >= bullish_points + 2:
        weather = "STORMY"
        description = "Downside signals are dominating."
    elif volatility >= 25:
        weather = "WINDY"
        description = "Mixed signals with elevated volatility."
    else:
        weather = "CLOUDY"
        description = "Signals are mixed."

    return {
        "condition": weather,
        "description": description
    }


# ============================================================
# FINANCIAL DNA
# ============================================================

def financial_dna(company):
    """
    Generate an interpretable company profile.
    """

    growth = company.get(
        "revenue_growth"
    )

    margin = company.get(
        "profit_margin"
    )

    roe = company.get(
        "return_on_equity"
    )

    pe = company.get(
        "pe_ratio"
    )

    if growth is not None and growth >= 15:
        growth_type = "HIGH GROWTH"
    elif growth is not None and growth >= 5:
        growth_type = "STEADY GROWTH"
    elif growth is not None and growth < 0:
        growth_type = "CONTRACTING"
    else:
        growth_type = "UNKNOWN"

    if margin is not None and margin >= 20:
        profitability = "HIGH MARGIN"
    elif margin is not None and margin >= 10:
        profitability = "PROFITABLE"
    elif margin is not None and margin < 0:
        profitability = "LOSS MAKING"
    else:
        profitability = "MIXED"

    if pe is not None and pe < 20:
        valuation = "VALUE"
    elif pe is not None and pe < 35:
        valuation = "FAIRLY VALUED"
    elif pe is not None:
        valuation = "PREMIUM"
    else:
        valuation = "UNKNOWN"

    return {
        "growth": growth_type,
        "profitability": profitability,
        "valuation": valuation,
        "return_profile": (
            "Strong capital efficiency"
            if roe is not None and roe >= 20
            else "Moderate capital efficiency"
            if roe is not None
            else "Insufficient data"
        )
    }


# ============================================================
# RED FLAGS
# ============================================================

def generate_red_flags(
    company,
    rsi,
    volatility,
    max_drawdown,
    sentiment_score
):
    flags = []

    pe = company.get("pe_ratio")
    revenue_growth = company.get(
        "revenue_growth"
    )
    profit_margin = company.get(
        "profit_margin"
    )

    if pe is not None and pe > 50:
        flags.append({
            "severity": "HIGH",
            "title": "High valuation",
            "detail": f"P/E ratio is {pe}, which indicates a premium valuation."
        })

    if revenue_growth is not None and revenue_growth < 0:
        flags.append({
            "severity": "HIGH",
            "title": "Revenue contraction",
            "detail": f"Revenue growth is {revenue_growth}%."
        })

    if profit_margin is not None and profit_margin < 0:
        flags.append({
            "severity": "HIGH",
            "title": "Negative profit margin",
            "detail": f"Reported profit margin is {profit_margin}%."
        })

    if rsi >= 70:
        flags.append({
            "severity": "MEDIUM",
            "title": "Potentially overbought",
            "detail": f"RSI is {rsi}."
        })

    if volatility >= 40:
        flags.append({
            "severity": "HIGH",
            "title": "High volatility",
            "detail": f"Annualized realized volatility is {volatility}%."
        })

    if max_drawdown <= -30:
        flags.append({
            "severity": "HIGH",
            "title": "Large historical drawdown",
            "detail": f"Maximum observed drawdown is {max_drawdown}%."
        })

    if sentiment_score <= 30:
        flags.append({
            "severity": "MEDIUM",
            "title": "Negative news sentiment",
            "detail": f"Recent news sentiment score is {sentiment_score}/100."
        })

    if not flags:
        flags.append({
            "severity": "LOW",
            "title": "No major automated red flags",
            "detail": "The current rule-based risk scan found no major warning."
        })

    return flags


# ============================================================
# SCENARIO ENGINE
# ============================================================

def build_scenarios(
    price,
    volatility,
    trend,
    sentiment_score
):
    """
    Bull/Base/Bear scenario model.
    """

    volatility_factor = max(
        0.05,
        min(
            0.30,
            volatility / 100
        )
    )

    if trend == "Bullish":
        bull_return = 0.15 + volatility_factor
        bear_return = -0.10 - volatility_factor / 2
    else:
        bull_return = 0.10 + volatility_factor / 2
        bear_return = -0.15 - volatility_factor

    if sentiment_score >= 65:
        bull_return += 0.03
    elif sentiment_score <= 35:
        bear_return -= 0.03

    base_return = (
        bull_return +
        bear_return
    ) / 2

    scenarios = [
        {
            "name": "BULL",
            "probability": 30,
            "return_percent": round(
                bull_return * 100,
                2
            ),
            "target_price": round(
                price * (1 + bull_return),
                2
            ),
            "thesis": "Momentum, sentiment and business execution remain supportive."
        },
        {
            "name": "BASE",
            "probability": 45,
            "return_percent": round(
                base_return * 100,
                2
            ),
            "target_price": round(
                price * (1 + base_return),
                2
            ),
            "thesis": "Mixed signals produce a moderate expected outcome."
        },
        {
            "name": "BEAR",
            "probability": 25,
            "return_percent": round(
                bear_return * 100,
                2
            ),
            "target_price": round(
                price * (1 + bear_return),
                2
            ),
            "thesis": "Valuation, volatility or negative sentiment creates downside pressure."
        }
    ]

    return scenarios


# ============================================================
# SPIDER SENSE
# ============================================================

def calculate_spider_sense(
    technical_signal,
    fundamental_signal,
    sentiment_signal,
    technical_confidence,
    fundamental_confidence,
    sentiment_confidence,
    risk_level
):
    """
    Weighted multi-agent intelligence score.
    """

    signal_scores = {
        "BULLISH": 100,
        "NEUTRAL": 50,
        "BEARISH": 0
    }

    technical_score = signal_scores.get(
        technical_signal,
        50
    )

    fundamental_score = signal_scores.get(
        fundamental_signal,
        50
    )

    sentiment_score = signal_scores.get(
        sentiment_signal,
        50
    )

    base_score = (
        technical_score * 0.35 +
        fundamental_score * 0.35 +
        sentiment_score * 0.30
    )

    confidence_factor = (
        technical_confidence * 0.35 +
        fundamental_confidence * 0.35 +
        sentiment_confidence * 0.30
    )

    score = (
        base_score * 0.75 +
        confidence_factor * 100 * 0.25
    )

    if risk_level == "HIGH":
        score -= 8
    elif risk_level == "MODERATE":
        score -= 3

    score = max(
        0,
        min(100, score)
    )

    return round(score)


# ============================================================
# MAIN ANALYSIS ENDPOINT
# ============================================================

@app.get("/analyze/{symbol}")
def analyze_stock(
    symbol: str,
    profile: Optional[str] = Query(
        "Moderate"
    ),
    horizon: Optional[str] = Query(
        "Long-Term"
    ),
    capital: Optional[float] = Query(
        10000
    )
):

    start_time = time.time()

    ticker = symbol.upper().strip()

    if not ticker:
        raise HTTPException(
            status_code=400,
            detail="Ticker symbol is required."
        )

    allowed_profiles = [
        "Conservative",
        "Moderate",
        "Aggressive"
    ]

    if profile not in allowed_profiles:
        profile = "Moderate"

    if capital <= 0:
        capital = 10000

    # --------------------------------------------------------
    # 1. LOAD REAL MARKET DATA
    # --------------------------------------------------------

    is_degraded = False

    try:

        ticker_obj = yf.Ticker(ticker)

        hist = ticker_obj.history(
            period="3mo",
            auto_adjust=False
        )

        if hist.empty:
            raise ValueError(
                "No market data returned."
            )

        close_series = (
            hist["Close"]
            .dropna()
        )

        prices = [
            float(value)
            for value in close_series.tolist()
        ]

        if not prices:
            raise ValueError(
                "No closing prices available."
            )

        current_price = round(
            prices[-1],
            2
        )

        previous_price = (
            prices[-2]
            if len(prices) >= 2
            else prices[-1]
        )

        daily_change = 0.0

        if previous_price != 0:
            daily_change = (
                (current_price - previous_price)
                / previous_price
            ) * 100

        daily_change = round(
            daily_change,
            2
        )

        sma_20 = round(
            sum(prices[-20:]) /
            len(prices[-20:]),
            2
        )

        sma_series = calculate_sma_series(
            prices,
            20
        )

        chart_labels = [
            index.strftime("%b %d")
            for index in hist.index
        ]

        chart_prices = [
            round(float(value), 2)
            for value in prices
        ]

        rsi = calculate_rsi(
            prices
        )

        max_drawdown = calculate_max_drawdown(
            prices
        )

        volatility = calculate_volatility(
            prices
        )

    except Exception:

        # ----------------------------------------------------
        # SAFE FALLBACK
        # ----------------------------------------------------

        ticker_obj = yf.Ticker(ticker)

        current_price = 150.00

        daily_change = 0.0

        sma_20 = 145.00

        chart_labels = [
            "Fallback 1",
            "Fallback 2",
            "Fallback 3",
            "Fallback 4",
            "Fallback 5"
        ]

        chart_prices = [
            142.0,
            144.5,
            143.0,
            148.0,
            150.0
        ]

        sma_series = [
            145.0
            for _ in chart_prices
        ]

        rsi = 50.0

        max_drawdown = -5.0

        volatility = 20.0

        is_degraded = True

    # --------------------------------------------------------
    # 2. COMPANY INFORMATION
    # --------------------------------------------------------

    company = get_company_information(
        ticker_obj,
        ticker
    )

    # --------------------------------------------------------
    # 3. NEWS SENTIMENT
    # --------------------------------------------------------

    news_data = get_news_sentiment(
        ticker_obj
    )

    sentiment_score = news_data[
        "score"
    ]

    # --------------------------------------------------------
    # 4. BUILD AGENT INPUT
    # --------------------------------------------------------

    stock_for_agents = {
        "ticker": ticker,
        "price": current_price,
        "change": daily_change,
        "volume": (
            int(hist["Volume"].iloc[-1])
            if not is_degraded and "Volume" in hist.columns
            else 0
        ),
        "pe": company.get(
            "pe_ratio"
        ) or 0,
        "revenue_growth": company.get(
            "revenue_growth"
        ) or 0,
        "sentiment": (
            (sentiment_score - 50) / 50
        )
    }

    # --------------------------------------------------------
    # 5. RUN INDEPENDENT AGENTS
    # --------------------------------------------------------

    technical_result = technical_agent(
        stock_for_agents
    )

    fundamental_result = fundamental_agent(
        stock_for_agents
    )

    sentiment_result = sentiment_agent(
        stock_for_agents
    )

    # --------------------------------------------------------
    # 6. LEGACY TREND / VERDICT
    # --------------------------------------------------------

    price_signal = (
        "Bullish"
        if current_price > sma_20
        else "Bearish"
    )

    if profile == "Conservative":

        verdict = (
            "HOLD"
            if price_signal == "Bullish"
            else "SELL"
        )

        impact = (
            f"Conservative Profile: "
            f"Position sizing prioritizes drawdown control for {ticker}."
        )

    elif profile == "Aggressive":

        verdict = (
            "STRONG BUY"
            if price_signal == "Bullish"
            else "SHORT"
        )

        impact = (
            f"Aggressive Profile: "
            f"Position sizing allows greater exposure to {ticker}."
        )

    else:

        verdict = (
            "BUY"
            if price_signal == "Bullish"
            else "SELL"
        )

        impact = (
            f"Moderate Profile: "
            f"Balanced exposure to {ticker} based on current signals."
        )

    # --------------------------------------------------------
    # 7. RISK ENGINE
    # --------------------------------------------------------

    risk_engine = build_risk_engine(
        current_price,
        rsi,
        volatility,
        max_drawdown,
        profile,
        capital
    )

    risk_level = risk_engine[
        "risk_level"
    ]

    # --------------------------------------------------------
    # 8. MARKET WEATHER
    # --------------------------------------------------------

    weather = market_weather(
        price_signal,
        rsi,
        volatility,
        sentiment_score
    )

    # --------------------------------------------------------
    # 9. FINANCIAL DNA
    # --------------------------------------------------------

    dna = financial_dna(
        company
    )

    # --------------------------------------------------------
    # 10. RED FLAGS
    # --------------------------------------------------------

    red_flags = generate_red_flags(
        company,
        rsi,
        volatility,
        max_drawdown,
        sentiment_score
    )

    # --------------------------------------------------------
    # 11. SCENARIO ENGINE
    # --------------------------------------------------------

    scenarios = build_scenarios(
        current_price,
        volatility,
        price_signal,
        sentiment_score
    )

    # --------------------------------------------------------
    # 12. SPIDER SENSE
    # --------------------------------------------------------

    spider_sense = calculate_spider_sense(
        technical_result["signal"],
        fundamental_result["signal"],
        sentiment_result["signal"],
        technical_result["confidence"],
        fundamental_result["confidence"],
        sentiment_result["confidence"],
        risk_level
    )

    # --------------------------------------------------------
    # 13. TRUST SCORE
    # --------------------------------------------------------

    trust_score = 100

    if is_degraded:
        trust_score -= 25

    if not company.get("name"):
        trust_score -= 10

    if not news_data.get("articles"):
        trust_score -= 10

    trust_score = max(
        0,
        min(100, trust_score)
    )

    # --------------------------------------------------------
    # 14. SEC EDGAR
    # --------------------------------------------------------

    sec_data = get_sec_company_data(
        ticker
    )

    latest_filing = sec_data.get(
        "latest_filing"
    )

    if sec_data.get("available"):

        if latest_filing:

            retrieved_text = (
                f"SEC EDGAR record for "
                f"{company.get('name', ticker)}. "
                f"Latest major filing: "
                f"{latest_filing.get('form')} "
                f"filed on "
                f"{latest_filing.get('filing_date')}."
            )

        else:

            retrieved_text = (
                f"SEC EDGAR has a company record "
                f"for {company.get('name', ticker)}, "
                f"but no recent 10-K/10-Q/8-K filing "
                f"was identified."
            )

        sec_source = "SEC EDGAR"

    else:

        retrieved_text = (
            f"SEC filing lookup unavailable for {ticker}. "
            f"Company and market data are still available "
            f"from the market-data provider."
        )

        sec_source = "SEC EDGAR unavailable"

    # --------------------------------------------------------
    # 15. AGENT DEBATE
    # --------------------------------------------------------

    agent_votes = [
        technical_result["signal"],
        fundamental_result["signal"],
        sentiment_result["signal"]
    ]

    bullish_votes = agent_votes.count(
        "BULLISH"
    )

    bearish_votes = agent_votes.count(
        "BEARISH"
    )

    neutral_votes = agent_votes.count(
        "NEUTRAL"
    )

    if bullish_votes >= 2:
        consensus = "BULLISH"
    elif bearish_votes >= 2:
        consensus = "BEARISH"
    else:
        consensus = "MIXED"

    debate = [
        {
            "agent": "Technical Analyst",
            "stance": technical_result["signal"],
            "confidence": technical_result["confidence"],
            "argument": technical_result["reasoning"]
        },
        {
            "agent": "Fundamental Analyst",
            "stance": fundamental_result["signal"],
            "confidence": fundamental_result["confidence"],
            "argument": fundamental_result["reasoning"]
        },
        {
            "agent": "Sentiment Analyst",
            "stance": sentiment_result["signal"],
            "confidence": sentiment_result["confidence"],
            "argument": sentiment_result["reasoning"]
        }
    ]

    # --------------------------------------------------------
    # 16. DECISION PASSPORT
    # --------------------------------------------------------

    decision_passport = {
        "ticker": ticker,
        "company": company.get(
            "name",
            ticker
        ),
        "profile": profile,
        "horizon": horizon,
        "capital": capital,
        "verdict": verdict,
        "spider_sense": spider_sense,
        "trust_score": trust_score,
        "agent_consensus": consensus,
        "risk_level": risk_level,
        "timestamp": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }

    # --------------------------------------------------------
    # 17. PERFORMANCE
    # --------------------------------------------------------

    latency_ms = round(
        (time.time() - start_time) * 1000,
        2
    )

    # --------------------------------------------------------
    # 18. FINAL API RESPONSE
    # --------------------------------------------------------

    return {

        # ====================================================
        # BASIC
        # ====================================================

        "ticker": ticker,

        "profile_applied": profile,

        "horizon_applied": horizon,

        "capital_applied": capital,

        "is_degraded_data": is_degraded,

        # ====================================================
        # COMPANY
        # ====================================================

        "company": company,

        # ====================================================
        # MARKET DATA
        # ====================================================

        "market_data": {

            "current_price": current_price,

            "daily_change": daily_change,

            "sma_20": sma_20,

            "sma_series": sma_series,

            "trend": price_signal,

            "rsi": rsi,

            "volatility": volatility,

            "max_drawdown": max_drawdown,

            "chart_labels": chart_labels,

            "chart_prices": chart_prices
        },

        # ====================================================
        # NEWS
        # ====================================================

        "news": news_data,

        # ====================================================
        # SEC
        # ====================================================

        "sec": sec_data,

        "rag_corpus": {

            "retrieved_text": retrieved_text,

            "source": sec_source
        },

        # ====================================================
        # AGENTS
        # ====================================================

        "agents": {

            "technical": technical_result,

            "fundamental": fundamental_result,

            "sentiment": sentiment_result
        },

        # ====================================================
        # ANALYSIS
        # ====================================================

        "analysis": {

            "verdict": verdict,

            "confidence_score": spider_sense,

            "sentiment_score": sentiment_score,

            "risk_level": risk_level,

            "portfolio_impact": impact,

            "spider_sense": spider_sense,

            "trust_score": trust_score,

            "agent_consensus": consensus,

            "chain_of_analysis": [

                {
                    "agent": "Technical Agent",

                    "output": (
                        f"Current price "
                        f"${current_price} vs "
                        f"20-Day SMA "
                        f"${sma_20}. "
                        f"Trend: {price_signal}. "
                        f"RSI: {rsi}."
                    ),

                    "confidence": round(
                        technical_result["confidence"] * 100
                    ),

                    "source": (
                        "Yahoo Finance market data"
                        if not is_degraded
                        else "Fallback Engine"
                    )
                },

                {
                    "agent": "Sentiment Agent",

                    "output": (
                        f"Recent news sentiment "
                        f"score: {sentiment_score}/100."
                    ),

                    "confidence": round(
                        sentiment_result["confidence"] * 100
                    ),

                    "source": "Yahoo Finance news feed"
                },

                {
                    "agent": "Fundamental Agent",

                    "output": (
                        f"Revenue growth: "
                        f"{company.get('revenue_growth')}; "
                        f"P/E: "
                        f"{company.get('pe_ratio')}."
                    ),

                    "confidence": round(
                        fundamental_result["confidence"] * 100
                    ),

                    "source": "Yahoo Finance company data"
                },

                {
                    "agent": "SEC Evidence Agent",

                    "output": retrieved_text,

                    "confidence": 95
                    if sec_data.get("available")
                    else 40,

                    "source": sec_source
                },

                {
                    "agent": "Orchestrator Agent",

                    "output": (
                        f"Combined 3-agent signals "
                        f"using {profile} risk profile. "
                        f"Final decision: {verdict}."
                    ),

                    "confidence": spider_sense,

                    "source": "FinLens Consensus Engine"
                }
            ]
        },

        # ====================================================
        # RISK
        # ====================================================

        "risk_engine": risk_engine,

        # ====================================================
        # MARKET WEATHER
        # ====================================================

        "market_weather": weather,

        # ====================================================
        # FINANCIAL DNA
        # ====================================================

        "financial_dna": dna,

        # ====================================================
        # RED FLAGS
        # ====================================================

        "red_flags": red_flags,

        # ====================================================
        # SCENARIOS
        # ====================================================

        "scenarios": scenarios,

        # ====================================================
        # DEBATE
        # ====================================================

        "agent_debate": {

            "consensus": consensus,

            "bullish_votes": bullish_votes,

            "bearish_votes": bearish_votes,

            "neutral_votes": neutral_votes,

            "agents": debate
        },

        # ====================================================
        # DECISION PASSPORT
        # ====================================================

        "decision_passport": decision_passport,

        # ====================================================
        # DATA QUALITY
        # ====================================================

        "data_quality": {

            "market_data": (
                "LIVE"
                if not is_degraded
                else "FALLBACK"
            ),

            "company_data": (
                "AVAILABLE"
                if company.get("name")
                else "LIMITED"
            ),

            "news_data": (
                "AVAILABLE"
                if news_data.get("articles")
                else "LIMITED"
            ),

            "sec_data": (
                "VERIFIED"
                if sec_data.get("available")
                else "UNAVAILABLE"
            ),

            "trust_score": trust_score
        },

        # ====================================================
        # PERFORMANCE
        # ====================================================

        "performance_logs": {

            "latency_ms": latency_ms,

            "signal_accuracy": "Rule-based",

            "risk_concentration_score": round(
                risk_engine[
                    "recommended_allocation_percent"
                ] / 100,
                2
            )
        }
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "engine": "FinLens AI",
        "version": "2.0.0",
        "message": "Multi-Agent Financial Intelligence Engine"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )