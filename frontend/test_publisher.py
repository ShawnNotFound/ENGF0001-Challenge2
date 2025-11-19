#!/usr/bin/env python3
"""
Simple MQTT publisher that continuously pushes mock telemetry to the dashboard.
Publishes individual messages to temperature, pH, and stirring topics.
"""

import argparse
import json
import random
import signal
import sys
import time

try:
  import paho.mqtt.client as mqtt
except ImportError as exc:  # pragma: no cover - runtime guard
  raise SystemExit(
      "paho-mqtt is required. Install with `pip install paho-mqtt`."
  ) from exc


def build_parser():
  parser = argparse.ArgumentParser(description="Publish mock telemetry to MQTT broker.")
  parser.add_argument("--host", default="localhost", help="MQTT broker host (default: localhost)")
  parser.add_argument("--port", default=1883, type=int, help="MQTT broker TCP port (default: 1883)")
  parser.add_argument("--interval", default=1, type=float, help="Seconds between samples (default: 1)")
  parser.add_argument("--temp", default=30.0, type=float, help="Baseline temperature °C")
  parser.add_argument("--ph", default=5.0, type=float, help="Baseline pH")
  parser.add_argument("--stir", default=900.0, type=float, help="Baseline stirring PWM")
  return parser


def format_payload(value):
  return json.dumps({
      "value": value,
      "ts": int(time.time() * 1000)
  })


def main():
  args = build_parser().parse_args()
  client = mqtt.Client()
  client.connect(args.host, args.port, keepalive=30)
  client.loop_start()

  running = True

  def stop(*_):
    nonlocal running
    running = False

  signal.signal(signal.SIGINT, stop)
  signal.signal(signal.SIGTERM, stop)

  topics = {
      "bioreactor/temperature": lambda: round(random.gauss(args.temp, 0.15), 2),
      "bioreactor/ph": lambda: round(random.gauss(args.ph, 0.02), 3),
      "bioreactor/stirring_pwm": lambda: round(random.gauss(args.stir, 10), 1),
  }

  print(f"Publishing mock telemetry to {args.host}:{args.port} every {args.interval}s. Ctrl+C to stop.")
  try:
    while running:
      for topic, generator in topics.items():
        payload = format_payload(generator())
        client.publish(topic, payload)
      time.sleep(args.interval)
  finally:
    client.loop_stop()
    client.disconnect()
    print("Publisher stopped.")


if __name__ == "__main__":
  main()
