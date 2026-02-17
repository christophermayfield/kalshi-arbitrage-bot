from typing import Dict, List, Any, Optional
from datetime import datetime
import aiohttp
import json
from dataclasses import dataclass
from collections import defaultdict

from src.utils.logging_utils import get_logger
from src.utils.database import Database

logger = get_logger("sentiment_analyzer")


@dataclass
class SentimentScore:
    """Data class for sentiment scores"""

    source: str
    score: float  # -1 to 1 (negative to positive)
    confidence: float  # 0 to 1
    timestamp: str
    text: Optional[str] = None


class NewsSentimentAnalyzer:
    """News sentiment analysis using FinBERT and external APIs"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.cache_ttl = 3600  # 1 hour cache
        self.db = Database()
        self._cache: Dict[str, Dict[str, Any]] = {}

    async def fetch_finnhub_sentiment(
        self, query: str, sources: List[str] = None
    ) -> Optional[List[SentimentScore]]:
        """Fetch sentiment from Finnhub API"""
        if not self.api_key:
            logger.warning("Finnhub API key not provided")
            return None

        sources = sources or ["reuters", "bloomberg", "business-insider"]

        try:
            async with aiohttp.ClientSession() as session:
                url = "https://finnhub.io/api/news"
                params = {
                    "q": query,
                    "token": self.api_key,
                    "sources": ",".join(sources),
                    "minRelevanceScore": "0.5",
                }

                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_finnhub_news(data)
                    else:
                        logger.error(f"Finnhub API error: {response.status}")
                        return None

        except Exception as e:
            logger.error(f"Error fetching Finnhub sentiment: {e}")
            return None

    def _process_finnhub_news(self, news_data: List[Dict]) -> List[SentimentScore]:
        """Process Finnhub news data"""
        scores = []

        for article in news_data:
            if "sentiment" in article:
                # Finnhub provides pre-sentimented news
                sentiment_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
                score = sentiment_map.get(article["sentiment"], 0.0)

                scores.append(
                    SentimentScore(
                        source="finnhub",
                        score=score,
                        confidence=0.8,  # Finnhub confidence
                        timestamp=article.get(
                            "datetime", datetime.utcnow().isoformat()
                        ),
                        text=article.get("headline", ""),
                    )
                )

        return scores

    async def fetch_newsapi_sentiment(
        self, query: str, language: str = "en"
    ) -> Optional[List[SentimentScore]]:
        """Fetch sentiment from NewsAPI"""
        if not self.api_key:
            logger.warning("NewsAPI key not provided")
            return None

        try:
            async with aiohttp.ClientSession() as session:
                url = "https://newsapi.org/v2/everything"
                params = {
                    "q": query,
                    "language": language,
                    "apiKey": self.api_key,
                    "sortBy": "relevancy",
                    "pageSize": 10,
                }

                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_newsapi_articles(data)
                    else:
                        logger.error(f"NewsAPI error: {response.status}")
                        return None

        except Exception as e:
            logger.error(f"Error fetching NewsAPI sentiment: {e}")
            return None

    def _process_newsapi_articles(self, data: Dict) -> List[SentimentScore]:
        """Process NewsAPI articles using VADER for sentiment"""
        scores = []

        try:
            # Import here to avoid dependency issues
            from nltk.sentiment.vader import SentimentIntensityAnalyzer

            sia = SentimentIntensityAnalyzer()

            for article in data.get("articles", []):
                if article.get("title"):
                    # Use VADER for sentiment analysis
                    sentiment_dict = sia.polarity_scores(article["title"])
                    compound_score = sentiment_dict["compound"]

                    scores.append(
                        SentimentScore(
                            source="newsapi",
                            score=compound_score,
                            confidence=abs(
                                compound_score
                            ),  # Higher confidence for extreme scores
                            timestamp=datetime.utcnow().isoformat(),
                            text=article["title"],
                        )
                    )

        except ImportError:
            logger.error("NLTK/VADER not available, skipping NewsAPI sentiment")
        except Exception as e:
            logger.error(f"Error processing NewsAPI sentiment: {e}")

        return scores


class SocialSentimentAnalyzer:
    """Social media sentiment analysis"""

    def __init__(self, twitter_bearer_token: Optional[str] = None):
        self.twitter_bearer_token = twitter_bearer_token
        self.cache_ttl = 1800  # 30 minutes cache
        self.db = Database()

    async def fetch_twitter_sentiment(
        self, query: str, count: int = 100
    ) -> Optional[List[SentimentScore]]:
        """Fetch sentiment from Twitter API"""
        if not self.twitter_bearer_token:
            logger.warning("Twitter API key not provided")
            return None

        try:
            async with aiohttp.ClientSession() as session:
                # Twitter API v2 search
                url = "https://api.twitter.com/2/tweets/search/recent"
                params = {
                    "query": query,
                    "max_results": min(count, 100),
                    "tweet.fields": "created_at,public_metrics",
                }
                headers = {"Authorization": f"Bearer {self.twitter_bearer_token}"}

                async with session.get(
                    url, params=params, headers=headers, timeout=15
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_twitter_tweets(data)
                    else:
                        logger.error(f"Twitter API error: {response.status}")
                        return None

        except Exception as e:
            logger.error(f"Error fetching Twitter sentiment: {e}")
            return None

    def _process_twitter_tweets(self, data: Dict) -> List[SentimentScore]:
        """Process Twitter tweets using VADER"""
        scores = []

        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer

            sia = SentimentIntensityAnalyzer()

            for tweet in data.get("data", []):
                if "text" in tweet:
                    sentiment_dict = sia.polarity_scores(tweet["text"])
                    compound_score = sentiment_dict["compound"]

                    # Weight by engagement metrics
                    engagement = tweet.get("public_metrics", {}).get("retweet_count", 0)
                    engagement_weight = min(1.0, 1 + engagement * 0.1)

                    scores.append(
                        SentimentScore(
                            source="twitter",
                            score=compound_score,
                            confidence=abs(compound_score) * engagement_weight,
                            timestamp=tweet.get(
                                "created_at", datetime.utcnow().isoformat()
                            ),
                            text=tweet["text"],
                        )
                    )

        except ImportError:
            logger.error("NLTK/VADER not available, skipping Twitter sentiment")
        except Exception as e:
            logger.error(f"Error processing Twitter sentiment: {e}")

        return scores

    async def fetch_stocktwits_sentiment(
        self, symbol: str
    ) -> Optional[List[SentimentScore]]:
        """Fetch sentiment from StockTwits"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
                params = {"limit": 50}

                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_stocktwits_messages(data)
                    else:
                        logger.error(f"StockTwits API error: {response.status}")
                        return None

        except Exception as e:
            logger.error(f"Error fetching StockTwits sentiment: {e}")
            return None

    def _process_stocktwits_messages(self, data: Dict) -> List[SentimentScore]:
        """Process StockTwits messages"""
        scores = []

        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer

            sia = SentimentIntensityAnalyzer()

            for message in data.get("messages", []):
                if "body" in message:
                    sentiment_dict = sia.polarity_scores(message["body"])
                    compound_score = sentiment_dict["compound"]

                    # Bullish/Bearish sentiment is often more explicit
                    if "sentiment" in message:
                        sentiment = message["sentiment"].get("basic", "")
                        if sentiment.lower() == "bullish":
                            compound_score = max(compound_score, 0.5)
                        elif sentiment.lower() == "bearish":
                            compound_score = min(compound_score, -0.5)

                    scores.append(
                        SentimentScore(
                            source="stocktwits",
                            score=compound_score,
                            confidence=abs(compound_score),
                            timestamp=message.get(
                                "created_at", datetime.utcnow().isoformat()
                            ),
                            text=message["body"],
                        )
                    )

        except ImportError:
            logger.error("NLTK/VADER not available, skipping StockTwits sentiment")
        except Exception as e:
            logger.error(f"Error processing StockTwits sentiment: {e}")

        return scores


class SentimentAggregator:
    """Aggregates sentiment from multiple sources"""

    def __init__(self):
        self.news_analyzer = NewsSentimentAnalyzer()
        self.social_analyzer = SocialSentimentAnalyzer()
        self.db = Database()

    async def get_market_sentiment(
        self, market_id: str, event_keywords: List[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get aggregated sentiment for a market"""
        event_keywords = event_keywords or []
        query = " ".join([market_id] + event_keywords)

        all_scores = []

        # Fetch news sentiment
        if self.news_analyzer.api_key:
            finnhub_scores = await self.news_analyzer.fetch_finnhub_sentiment(query)
            if finnhub_scores:
                all_scores.extend(finnhub_scores)

            newsapi_scores = await self.news_analyzer.fetch_newsapi_sentiment(query)
            if newsapi_scores:
                all_scores.extend(newsapi_scores)

        # Fetch social sentiment
        if self.social_analyzer.twitter_bearer_token:
            twitter_scores = await self.social_analyzer.fetch_twitter_sentiment(
                query, count=50
            )
            if twitter_scores:
                all_scores.extend(twitter_scores)

        # Use market symbol for StockTwits if available (extract from market_id)
        symbol = self._extract_symbol(market_id)
        if symbol:
            stocktwits_scores = await self.social_analyzer.fetch_stocktwits_sentiment(
                symbol
            )
            if stocktwits_scores:
                all_scores.extend(stocktwits_scores)

        if not all_scores:
            return None

        # Aggregate scores
        aggregated = self._aggregate_scores(all_scores)

        # Save to database
        self._save_sentiment_data(market_id, all_scores, aggregated)

        return aggregated

    def _extract_symbol(self, market_id: str) -> Optional[str]:
        """Extract symbol from market ID for StockTwits"""
        # This is a simple extraction - you may need to customize based on your market ID format
        if len(market_id) > 3 and not market_id.startswith("market_"):
            return market_id.upper()
        return None

    def _aggregate_scores(self, scores: List[SentimentScore]) -> Dict[str, Any]:
        """Aggregate sentiment scores from multiple sources"""
        if not scores:
            return {}

        # Group by source
        by_source = defaultdict(list)
        for score in scores:
            by_source[score.source].append(score)

        # Calculate weighted average (recent sources get higher weight)
        now = datetime.utcnow()
        total_weight = 0
        weighted_score = 0
        weighted_confidence = 0

        for source, source_scores in by_source.items():
            for score in source_scores:
                # Weight by recency (newer = higher weight)
                time_diff = (
                    now - datetime.fromisoformat(score.timestamp.replace("Z", "+00:00"))
                ).total_seconds()
                recency_weight = max(0.1, 1 - time_diff / 3600)  # Decay over hour

                # Weight by source reliability
                source_weights = {
                    "finnhub": 0.4,
                    "newsapi": 0.3,
                    "twitter": 0.2,
                    "stocktwits": 0.1,
                }
                source_weight = source_weights.get(source, 0.25)

                total_weight += recency_weight * source_weight
                weighted_score += score.score * recency_weight * source_weight
                weighted_confidence += score.confidence * recency_weight * source_weight

        if total_weight > 0:
            final_score = weighted_score / total_weight
            final_confidence = weighted_confidence / total_weight
        else:
            final_score = 0
            final_confidence = 0

        return {
            "market_id": scores[0].source if scores else "unknown",
            "sentiment_score": final_score,
            "confidence": final_confidence,
            "timestamp": datetime.utcnow().isoformat(),
            "source_count": len(by_source),
            "total_articles": len(scores),
            "source_breakdown": {
                source: len(source_scores)
                for source, source_scores in by_source.items()
            },
            "recent_articles": [
                {
                    "source": score.source,
                    "score": score.score,
                    "text": score.text[:100] + "..."
                    if score.text and len(score.text) > 100
                    else score.text,
                    "timestamp": score.timestamp,
                }
                for score in sorted(scores, key=lambda x: x.timestamp, reverse=True)[:5]
            ],
        }

    def _save_sentiment_data(
        self, market_id: str, scores: List[SentimentScore], aggregated: Dict[str, Any]
    ) -> None:
        """Save sentiment data to database"""
        try:
            query = """
                INSERT INTO sentiment_data (market_id, sentiment_score, confidence, source_count, total_articles, data)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            self.db.execute(
                query,
                (
                    market_id,
                    aggregated["sentiment_score"],
                    aggregated["confidence"],
                    aggregated["source_count"],
                    aggregated["total_articles"],
                    json.dumps(aggregated),
                ),
            )
        except Exception as e:
            logger.error(f"Error saving sentiment data for market {market_id}: {e}")

    def get_historical_sentiment(
        self, market_id: str, hours: int = 24
    ) -> Optional[List[Dict[str, Any]]]:
        """Get historical sentiment data for a market"""
        try:
            query = """
                SELECT sentiment_score, confidence, timestamp, source_count, total_articles
                FROM sentiment_data 
                WHERE market_id = ? 
                AND timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp DESC
            """.format(hours)

            return self.db.query(query, (market_id,))

        except Exception as e:
            logger.error(
                f"Error retrieving historical sentiment for market {market_id}: {e}"
            )
            return None


async def get_market_sentiment_signal(
    market_id: str, event_keywords: List[str] = None
) -> Dict[str, Any]:
    """
    Get sentiment signal for a market
    Returns signal indicating sentiment influence on trading decisions
    """
    try:
        aggregator = SentimentAggregator()
        sentiment_data = await aggregator.get_market_sentiment(
            market_id, event_keywords
        )

        if not sentiment_data:
            return {
                "signal": "neutral",
                "reason": "No sentiment data available",
                "confidence": 0,
            }

        sentiment_score = sentiment_data["sentiment_score"]
        confidence = sentiment_data["confidence"]

        # Determine sentiment signal
        if sentiment_score > 0.3:
            signal = "bullish"
            reason = f"Positive sentiment detected (score: {sentiment_score:.3f})"
        elif sentiment_score < -0.3:
            signal = "bearish"
            reason = f"Negative sentiment detected (score: {sentiment_score:.3f})"
        else:
            signal = "neutral"
            reason = f"Neutral sentiment (score: {sentiment_score:.3f})"

        return {
            "signal": signal,
            "reason": reason,
            "confidence": confidence,
            "sentiment_score": sentiment_score,
            "market_id": market_id,
            "timestamp": datetime.utcnow().isoformat(),
            "source_count": sentiment_data.get("source_count", 0),
            "total_articles": sentiment_data.get("total_articles", 0),
        }

    except Exception as e:
        logger.error(f"Error generating sentiment signal for market {market_id}: {e}")
        return {
            "signal": "neutral",
            "reason": f"Error generating sentiment signal: {str(e)}",
            "confidence": 0,
        }
