from __future__ import annotations

import logging
import select
import threading
import time
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class EventType(IntEnum):
    READ = auto()
    WRITE = auto()
    EXCEPT = auto()


EventHandler = Callable[[int, EventType, Any], None]


@dataclass
class Event:
    event_type: EventType
    handler: EventHandler
    data: Any = None


class EventLoop:
    def __init__(self, name: str, poll_interval: float = 1.0, max_events: int = 1024):
        self.name = name
        self._running = False
        self._poll_interval = poll_interval
        self._max_events = max_events

        self._events: Dict[EventType, Dict[int, Event]] = {
            EventType.READ: {},
            EventType.WRITE: {},
            EventType.EXCEPT: {},
        }

        self._lock = threading.RLock()

    def _handle_event(self, fds: list[int], event_type: EventType):
        for fd in fds:
            with self._lock:
                if fd in self._events[event_type]:
                    event = self._events[event_type][fd]
                    try:
                        event.handler(fd, event_type, event.data)
                    except Exception as e:
                        logger.error(
                            f"EventLoop {self.name} handle {event_type.name} event error (fd={fd}): {e}"
                        )

    def _events_count(self) -> int:
        return sum(len(self._events[event_type]) for event_type in EventType)

    def run(self) -> None:
        self._running = True

        try:
            while self._running:
                with self._lock:
                    read_fds = list(self._events[EventType.READ].keys())
                    write_fds = list(self._events[EventType.WRITE].keys())
                    except_fds = list(self._events[EventType.EXCEPT].keys())

                try:
                    readable, writable, exceptional = select.select(
                        read_fds, write_fds, except_fds, self._poll_interval
                    )

                    self._handle_event(readable, EventType.READ)
                    self._handle_event(writable, EventType.WRITE)
                    self._handle_event(exceptional, EventType.EXCEPT)

                except (ValueError, OSError) as e:
                    if self._running:
                        logger.error(f"EventLoop {self.name} select error: {e}")
                        time.sleep(self._poll_interval)
                except Exception as e:
                    logger.error(f"EventLoop {self.name} exception: {e}")

        except Exception as e:
            logger.error(f"EventLoop {self.name} exception: {e}")
        finally:
            logger.debug(f"EventLoop {self.name} exit")

    def stop(self) -> None:
        if not self._running:
            return

        with self._lock:
            self._running = False
            for event_type in EventType:
                self._events[event_type].clear()

    def is_running(self) -> bool:
        return self._running

    def fd_is_registered(self, fd: int, event_type: Optional[EventType] = None) -> bool:
        with self._lock:
            if event_type is None:
                return (
                    fd in self._events[EventType.READ]
                    or fd in self._events[EventType.WRITE]
                    or fd in self._events[EventType.EXCEPT]
                )
            else:
                return fd in self._events[event_type]

    def register_fd(
        self, fd: int, event_type: EventType, handler: EventHandler, data: Any = None
    ) -> None:
        if self.fd_is_registered(fd, event_type):
            raise ValueError(
                f"EventLoop {self.name} fd {fd} has already registered for event type {event_type}"
            )

        if self._events_count() >= self._max_events:
            raise ValueError(
                f"EventLoop {self.name} max events count reached: {self._max_events}"
            )

        with self._lock:
            event = Event(event_type, handler, data)
            self._events[event_type][fd] = event

    def unregister_fd(self, fd: int, event_type: Optional[EventType] = None) -> None:
        if not self.fd_is_registered(fd, event_type):
            return

        with self._lock:
            if event_type is None:
                for ev_type in EventType:
                    if fd in self._events[ev_type]:
                        del self._events[ev_type][fd]
            else:
                if fd in self._events[event_type]:
                    del self._events[event_type][fd]
