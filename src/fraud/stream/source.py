"""Pluggable message transport.

The scoring logic is identical whether messages arrive over an in-memory queue
(demo/tests) or a real Kafka topic (production). Kafka is imported lazily so
neither the tests nor the demo require a running broker.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator


class MessageSource(ABC):
    @abstractmethod
    def produce(self, message: dict) -> None: ...

    @abstractmethod
    def consume(self) -> Iterator[dict]: ...


class InMemorySource(MessageSource):
    """FIFO queue used for the demo and tests, no broker required."""

    def __init__(self):
        self._queue: deque[dict] = deque()

    def produce(self, message: dict) -> None:
        self._queue.append(message)

    def consume(self) -> Iterator[dict]:
        while self._queue:
            yield self._queue.popleft()


class KafkaSource(MessageSource):
    """Real Kafka transport (via `docker compose --profile kafka up`)."""

    def __init__(self, bootstrap_servers: str, topic: str, group_id: str = "fraud-scorer"):
        from kafka import KafkaConsumer, KafkaProducer  # lazy: only needed for real Kafka

        self.topic = topic
        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        self._consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            auto_offset_reset="earliest",
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
            consumer_timeout_ms=10000,
        )

    def produce(self, message: dict) -> None:
        self._producer.send(self.topic, message)
        self._producer.flush()

    def consume(self) -> Iterator[dict]:
        for message in self._consumer:
            yield message.value
