# MQTT Dashboard

Small demo to monitor temperature, pH, and stirring PWM over time and publish new targets. It includes an MQTT broker (TCP + WebSocket) so other devices can subscribe to the same topics.

## Run

```bash
cd frontend
npm install
npm start
# open http://localhost:3000
```

Environment toggles:

- `HTTP_PORT` – serve UI and WebSocket MQTT on this port (default `3000`).
- `MQTT_PORT` – raw MQTT broker port (default `1883`).
- `MOCK_DATA=true` – publish synthetic telemetry every 2 seconds so the UI is not empty.

## MQTT Topics

- `bioreactor/temperature` – JSON `{ "value": <number>, "ts": <ms> }`
- `bioreactor/ph` – JSON `{ "value": <number>, "ts": <ms> }`
- `bioreactor/stirring_pwm` – JSON `{ "value": <number>, "ts": <ms> }`
- `bioreactor/telemetry` – Combined payload published by the mock generator.
- `bioreactor/anomaly` – Aggregated stats for anomaly detection (mean/min/max + setpoints), example payload:

  ```json
  {
    "temperature_C": { "mean": 35.75, "min": 35.69, "max": 35.79 },
    "pH": { "mean": 5.13, "min": 5.04, "max": 5.19 },
    "rpm": { "mean": 1004.88, "min": 982.89, "max": 1028.95 },
    "setpoints": { "temperature_C": 30.0, "pH": 5.0, "rpm": 1000.0 }
  }
  ```
- The server automatically maintains this topic by buffering the latest samples from the three telemetry feeds above (needs at least one reading per metric before it emits) and publishes at ~1 Hz.
- `bioreactor/anomaly/status` – Alerts coming from the Python anomaly detector. Payload example:

  ```json
  {
    "timestamp": 1700000000000,
    "temperature": { "z": 0.4, "level": 0 },
    "ph": { "z": 2.3, "level": 1 },
    "stirring_pwm": { "z": 3.5, "level": 2 },
    "anomalies": ["pH", "stirring speed"],
    "confirmed": true,
    "message": "Status: pH, stirring speed"
  }
  ```

  `level`: `0` normal, `1` caution (|z| > 2), `2` critical (|z| > 3). When `confirmed` is true the UI surfaces a warning banner.
- `bioreactor/target/{temperature|ph|stirring_pwm}` – Setpoints published from the UI (echoed back on `/echo` so all clients stay in sync).

Other devices can connect either via `mqtt://localhost:1883` (TCP) or `ws://<host>:<HTTP_PORT>` (WebSocket) to publish/subscribe.

### Historical charts

The dashboard keeps a rolling history (up to one week) of incoming telemetry in `localStorage`. Use the buttons above the charts to switch between views:

- `5 min` – last 5 minutes (default)
- `1 h` – last hour
- `24 h` – last day
- `All` – entire retained history

Your selection persists per browser, so refreshing or reopening keeps the chosen window. Historical data stays local to your browser and is not sent back to the server.

### Mock publisher

Need continuous telemetry without hardware? Install the Python dependency and run the helper script:

```bash
pip install paho-mqtt
cd frontend
python test_publisher.py --host localhost --port 1883 --interval 1.0
```

Flags let you customize baseline values: `--temp`, `--ph`, `--stir` (PWM). Stop with `Ctrl+C`.

## Publish / Subscribe cheat sheet

You can interact with the broker two ways:

1) **TCP MQTT** (good for embedded/CLI): `mqtt://localhost:1883`  
2) **WebSocket MQTT** (good for browser apps): `ws://localhost:3000`

### Subscribing (pull data)

- **CLI (using `npx mqtt-cli`)**

  ```bash
  npx mqtt sub -t bioreactor/anomaly -h localhost -p 1883
  npx mqtt sub -t bioreactor/temperature -h localhost -p 1883
  ```

- **Python (paho-mqtt)**

  ```python
  import json, paho.mqtt.client as mqtt

  def on_message(client, userdata, msg):
      print(msg.topic, json.loads(msg.payload))

  c = mqtt.Client()
  c.connect("localhost", 1883)
  c.subscribe("bioreactor/anomaly")
  c.subscribe("bioreactor/temperature")
  c.loop_forever()
  ```

- **Browser/JS (MQTT over WebSocket)**

  ```js
  import mqtt from "mqtt";
  const client = mqtt.connect("ws://localhost:3000");
  client.on("connect", () => client.subscribe("bioreactor/anomaly"));
  client.on("message", (topic, payload) => console.log(topic, payload.toString()));
  ```

### Publishing (push data)

Targets (setpoints) go to `bioreactor/target/{temperature|ph|stirring_pwm}` with JSON payload `{"value": number, "ts": <ms>}`:

- **CLI**

  ```bash
  npx mqtt pub -t bioreactor/target/temperature -h localhost -p 1883 -m '{"value":32,"ts":'$(date +%s%3N)'}'
  npx mqtt pub -t bioreactor/target/ph -h localhost -p 1883 -m '{"value":5.2,"ts":'$(date +%s%3N)'}'
  ```

- **Python**

  ```python
  import time, json, paho.mqtt.client as mqtt
  c = mqtt.Client(); c.connect("localhost", 1883)
  payload = json.dumps({"value": 32.0, "ts": int(time.time()*1000)})
  c.publish("bioreactor/target/temperature", payload)
  c.loop(1)
  ```

- **WebSocket (browser/JS)**

  ```js
  const client = mqtt.connect("ws://localhost:3000");
  client.on("connect", () => {
    client.publish("bioreactor/target/stirring_pwm", JSON.stringify({ value: 70, ts: Date.now() }));
  });
  ```

### Suggested wiring

- Controllers/edge devices: subscribe to `bioreactor/target/#`, publish telemetry to `bioreactor/temperature`, `bioreactor/ph`, `bioreactor/stirring_pwm`, and optionally `bioreactor/anomaly`.
- Dashboard (this UI): subscribes to telemetry + anomaly and publishes targets from the form.
