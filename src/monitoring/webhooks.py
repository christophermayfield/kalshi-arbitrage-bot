import json
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import httpx

from src.utils.logging_utils import get_logger

logger = get_logger("webhooks")


class WebhookProvider(str, Enum):
    SLACK = "slack"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    CUSTOM = "custom"


@dataclass
class WebhookMessage:
    title: str
    message: str
    level: str = "info"
    timestamp: str = None
    fields: Dict[str, str] = None
    color: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
        if self.fields is None:
            self.fields = {}


class WebhookHandler:
    def __init__(self):
        self.clients: Dict[WebhookProvider, Any] = {}
        self._default_provider = None

    def register_slack(self, webhook_url: str) -> None:
        self.clients[WebhookProvider.SLACK] = SlackWebhookClient(webhook_url)
        if self._default_provider is None:
            self._default_provider = WebhookProvider.SLACK

    def register_discord(self, webhook_url: str) -> None:
        self.clients[WebhookProvider.DISCORD] = DiscordWebhookClient(webhook_url)
        if self._default_provider is None:
            self._default_provider = WebhookProvider.DISCORD

    def register_telegram(self, token: str, chat_id: str) -> None:
        self.clients[WebhookProvider.TELEGRAM] = TelegramWebhookClient(token, chat_id)
        if self._default_provider is None:
            self._default_provider = WebhookProvider.TELEGRAM

    def register_custom(self, name: str, webhook_url: str, headers: Dict[str, str] = None) -> None:
        self.clients[f"custom_{name}"] = CustomWebhookClient(webhook_url, headers)

    async def send(self, message: WebhookMessage, providers: List[WebhookProvider] = None) -> Dict[WebhookProvider, bool]:
        if providers is None:
            providers = list(self.clients.keys())

        results = {}
        for provider in providers:
            if provider in self.clients:
                try:
                    success = await self.clients[provider].send(message)
                    results[provider] = success
                except Exception as e:
                    logger.error(f"Failed to send {provider} webhook: {e}")
                    results[provider] = False
        return results

    async def send_opportunity_alert(
        self,
        opportunity_id: str,
        profit_cents: int,
        confidence: float,
        market_id: str
    ) -> Dict[WebhookProvider, bool]:
        message = WebhookMessage(
            title="Arbitrage Opportunity Detected",
            message=f"Found opportunity {opportunity_id} on {market_id}",
            level="info",
            fields={
                "Opportunity ID": opportunity_id,
                "Profit": f"${profit_cents / 100:.2f}",
                "Confidence": f"{confidence:.1%}",
                "Market": market_id
            },
            color="#10B981"
        )
        return await self.send(message)

    async def send_execution_alert(
        self,
        opportunity_id: str,
        success: bool,
        profit_cents: int,
        error: Optional[str] = None
    ) -> Dict[WebhookProvider, bool]:
        level = "success" if success else "error"
        color = "#10B981" if success else "#EF4444"

        message = WebhookMessage(
            title="Arbitrage Execution Result",
            message=f"Execution of {opportunity_id}: {'Success' if success else 'Failed'}",
            level=level,
            fields={
                "Opportunity ID": opportunity_id,
                "Result": "Success" if success else "Failed",
                "Profit": f"${profit_cents / 100:.2f}" if profit_cents else "N/A",
                "Error": error if error else "None"
            },
            color=color
        )
        return await self.send(message)

    async def send_error_alert(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[WebhookProvider, bool]:
        message = WebhookMessage(
            title=f"Bot Error: {error_type}",
            message=error_message,
            level="error",
            fields={
                "Error Type": error_type,
                "Message": error_message,
                "Context": json.dumps(context) if context else "N/A"
            },
            color="#EF4444"
        )
        return await self.send(message)

    async def send_daily_summary(
        self,
        total_trades: int,
        winning_trades: int,
        total_profit: int,
        current_balance: int
    ) -> Dict[WebhookProvider, bool]:
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        message = WebhookMessage(
            title="Daily Trading Summary",
            message="Today's performance summary",
            level="info",
            fields={
                "Total Trades": str(total_trades),
                "Winning Trades": str(winning_trades),
                "Win Rate": f"{win_rate:.1f}%",
                "Total Profit": f"${total_profit / 100:.2f}",
                "Current Balance": f"${current_balance / 100:.2f}"
            },
            color="#3B82F6"
        )
        return await self.send(message)


class SlackWebhookClient:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    async def send(self, message: WebhookMessage) -> bool:
        payload = {
            "attachments": [
                {
                    "color": message.color or "#6B7280",
                    "title": message.title,
                    "text": message.message,
                    "fields": [
                        {"title": k, "value": v, "short": True}
                        for k, v in message.fields.items()
                    ],
                    "footer": "Kalshi Arbitrage Bot",
                    "ts": int(datetime.fromisoformat(message.timestamp).timestamp())
                }
            ]
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(self.webhook_url, json=payload)
            return response.status_code == 200


class DiscordWebhookClient:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    async def send(self, message: WebhookMessage) -> bool:
        color_map = {
            "info": 0x3B82F6,
            "success": 0x10B981,
            "warning": 0xF59E0B,
            "error": 0xEF4444
        }
        color = color_map.get(message.level, 0x6B7280)

        embed = {
            "title": message.title,
            "description": message.message,
            "color": color,
            "fields": [
                {"name": k, "value": v, "inline": True}
                for k, v in message.fields.items()
            ],
            "timestamp": message.timestamp
        }

        payload = {"embeds": [embed]}

        async with httpx.AsyncClient() as client:
            response = await client.post(self.webhook_url, json=payload)
            return response.status_code == 204


class TelegramWebhookClient:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{token}"

    async def send(self, message: WebhookMessage) -> bool:
        text = f"*{message.title}*\n\n{message.message}\n\n"
        text += "\n".join(f"*{k}*: {v}" for k, v in message.fields.items())

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_url}/sendMessage",
                json=payload
            )
            return response.status_code == 200


class CustomWebhookClient:
    def __init__(self, webhook_url: str, headers: Dict[str, str] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {}

    async def send(self, message: WebhookMessage) -> bool:
        payload = {
            "title": message.title,
            "message": message.message,
            "level": message.level,
            "timestamp": message.timestamp,
            "fields": message.fields
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.webhook_url,
                json=payload,
                headers=self.headers
            )
            return response.status_code in (200, 201, 204)


def create_webhook_handler_from_config(config: Dict[str, Any]) -> WebhookHandler:
    handler = WebhookHandler()

    notifications = config.get('notifications', {})

    if notifications.get('slack', {}).get('webhook_url'):
        handler.register_slack(notifications['slack']['webhook_url'])

    if notifications.get('discord', {}).get('webhook_url'):
        handler.register_discord(notifications['discord']['webhook_url'])

    if notifications.get('telegram', {}).get('token') and notifications.get('telegram', {}).get('chat_id'):
        handler.register_telegram(
            notifications['telegram']['token'],
            notifications['telegram']['chat_id']
        )

    return handler
