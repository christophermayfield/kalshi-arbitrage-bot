"""
Load tests for the Kalshi Arbitrage Bot.

Run with: locust -f tests/load/locustfile.py --host=http://localhost:8001
"""

from locust import HttpUser, task, between, events
from locust.runners import MasterRunner


class APIUser(HttpUser):
    wait_time = between(1, 5)
    host = "http://localhost:8001"

    @task(10)
    def health_check(self):
        with self.client.get("/health", name="Health Check", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    @task(5)
    def get_portfolio(self):
        with self.client.get("/portfolio", name="Get Portfolio", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    @task(3)
    def get_positions(self):
        with self.client.get("/positions", name="Get Positions", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    @task(2)
    def get_metrics(self):
        with self.client.get("/metrics", name="Get Metrics", catch_response=True) as response:
            if response.status_code == 200:
                response.success()

    @task(1)
    def get_config(self):
        with self.client.get("/config", name="Get Config", catch_response=True) as response:
            if response.status_code == 200:
                response.success()

    @task(1)
    def circuit_breakers(self):
        with self.client.get("/circuit-breakers", name="Get Circuit Breakers", catch_response=True) as response:
            if response.status_code == 200:
                response.success()


class BotControlUser(HttpUser):
    wait_time = between(10, 30)
    host = "http://localhost:8001"

    @task
    def start_bot(self):
        with self.client.post("/start", name="Start Bot", catch_response=True) as response:
            if response.status_code == 200:
                response.success()

    @task
    def stop_bot(self):
        with self.client.post("/stop", name="Stop Bot", catch_response=True) as response:
            if response.status_code == 200:
                response.success()

    @task
    def reload_config(self):
        with self.client.post("/config/reload", name="Reload Config", catch_response=True) as response:
            if response.status_code == 200:
                response.success()


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    if isinstance(environment.runner, MasterRunner):
        environment.runner.register_message(
            "stats_request",
            lambda msg: print(f"Stats received: {msg.data}")
        )


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("Load test completed")


if __name__ == "__main__":
    pass
