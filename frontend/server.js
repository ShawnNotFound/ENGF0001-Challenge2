const path = require('path');
const net = require('net');
const http = require('http');
const express = require('express');
const aedes = require('aedes')();
const websocketStream = require('websocket-stream');

const HTTP_PORT = process.env.HTTP_PORT || 3000; // Serves UI + MQTT over WebSocket
const MQTT_TCP_PORT = process.env.MQTT_PORT || 1883; // Raw MQTT for other devices
const ENABLE_MOCK_DATA = process.env.MOCK_DATA === 'true';
const ANOMALY_TOPIC = 'bioreactor/anomaly';
const MAX_BUFFER = parseInt(process.env.AGGREGATOR_WINDOW || '12', 10);
const MIN_ANOMALY_INTERVAL_MS = 1000;
const TARGET_STATE_TOPIC = 'bioreactor/target/state';

const DEFAULT_SETPOINTS = { temperature_C: 30.0, pH: 5.0, rpm: 1000.0 };
const setpointsCache = { ...DEFAULT_SETPOINTS };
const telemetryBuffers = {
  temperature: [],
  ph: [],
  stirring_pwm: []
};
const TELEMETRY_TOPIC_MAP = {
  'bioreactor/temperature': 'temperature',
  'bioreactor/ph': 'ph',
  'bioreactor/stirring_pwm': 'stirring_pwm'
};
const TARGET_FIELD_MAP = {
  temperature: 'temperature_C',
  ph: 'pH',
  stirring_pwm: 'rpm'
};

const safeParsePayload = payload => {
  if (!payload) return {};
  const text = payload.toString();
  try {
    return JSON.parse(text);
  } catch (err) {
    const numeric = parseFloat(text);
    return Number.isNaN(numeric) ? { raw: text } : { value: numeric };
  }
};

const pushSample = (key, value, ts) => {
  const buffer = telemetryBuffers[key];
  if (!buffer) return;
  buffer.push({ value, ts });
  if (buffer.length > MAX_BUFFER) buffer.shift();
};

const statsFor = (key) => {
  const buffer = telemetryBuffers[key];
  if (!buffer || buffer.length === 0) return null;
  const values = buffer.map(entry => entry.value);
  const decimals = key === 'ph' ? 3 : 2;
  const sum = values.reduce((acc, v) => acc + v, 0);
  const mean = +(sum / values.length).toFixed(decimals);
  const min = +Math.min(...values).toFixed(decimals);
  const max = +Math.max(...values).toFixed(decimals);
  return { mean, min, max };
};

let lastAnomalyPublish = 0;
const publishAggregateSnapshot = () => {
  const tempStats = statsFor('temperature');
  const phStats = statsFor('ph');
  const rpmStats = statsFor('stirring_pwm');
  if (!tempStats || !phStats || !rpmStats) return;

  const now = Date.now();
  if (now - lastAnomalyPublish < MIN_ANOMALY_INTERVAL_MS) return;
  lastAnomalyPublish = now;

  const payload = {
    temperature_C: tempStats,
    pH: phStats,
    rpm: rpmStats,
    setpoints: { ...setpointsCache },
    faults: { last_active: [], counts: {} }
  };
  aedes.publish({ topic: ANOMALY_TOPIC, payload: JSON.stringify(payload) });
};

const publishSetpointState = () => {
  aedes.publish({ topic: TARGET_STATE_TOPIC, payload: JSON.stringify(setpointsCache), retain: true });
};

const app = express();
app.use(express.static(path.join(__dirname, 'public')));

// TCP MQTT broker (port 1883 by default)
const tcpServer = net.createServer(aedes.handle);
tcpServer.listen(MQTT_TCP_PORT, () => {
  console.log(`MQTT broker (TCP) listening on port ${MQTT_TCP_PORT}`);
});

// HTTP server for static assets and MQTT over WebSocket
const httpServer = http.createServer(app);
websocketStream.createServer({ server: httpServer }, aedes.handle);
httpServer.listen(HTTP_PORT, () => {
  console.log(`Dashboard + MQTT over WebSocket running at http://localhost:${HTTP_PORT}`);
});

// Optional in-process mock data generator so the UI is not empty when no devices publish.
if (ENABLE_MOCK_DATA) {
  console.log('MOCK_DATA enabled: publishing synthetic telemetry every 2s.');
  const publishMock = () => {
    const now = Date.now();
    const payload = {
      timestamp: now,
      temperature: +(29 + Math.sin(now / 30000) * 2 + Math.random() * 0.3).toFixed(2),
      ph: +(5 + Math.sin(now / 45000) * 0.4 + Math.random() * 0.05).toFixed(2),
      stirring: +(60 + Math.sin(now / 20000) * 15 + Math.random() * 2).toFixed(1)
    };

    [
      { topic: 'bioreactor/temperature', payload: JSON.stringify({ value: payload.temperature, ts: now }) },
      { topic: 'bioreactor/ph', payload: JSON.stringify({ value: payload.ph, ts: now }) },
      { topic: 'bioreactor/stirring_pwm', payload: JSON.stringify({ value: payload.stirring, ts: now }) },
      { topic: 'bioreactor/telemetry', payload: JSON.stringify(payload) }
    ].forEach(pkt => aedes.publish(pkt));
  };
  setInterval(publishMock, 2000);
}

// Echo setpoint changes so other subscribers see them and aggregate telemetry samples.
aedes.on('publish', (packet, client) => {
  if (!packet?.topic) return;
  if (packet.topic === ANOMALY_TOPIC) return;

  const isTargetChange =
    packet.topic.startsWith('bioreactor/target/') && !packet.topic.endsWith('/echo');
  if (isTargetChange) {
    const targetName = packet.topic.split('/')[2];
    const parsed = safeParsePayload(packet.payload);
    const field = TARGET_FIELD_MAP[targetName];
    const numericValue = parsed && typeof parsed.value !== 'undefined'
      ? Number(parsed.value)
      : Number(parsed[targetName]);
    if (field && !Number.isNaN(numericValue)) {
      setpointsCache[field] = numericValue;
      publishSetpointState();
    }
    // Fan out an echo topic so UI stays in sync even when a different client sets targets.
    const echoTopic = `${packet.topic}/echo`;
    aedes.publish({ topic: echoTopic, payload: packet.payload });
    return;
  }

  const telemetryKey = TELEMETRY_TOPIC_MAP[packet.topic];
  if (telemetryKey) {
    const parsed = safeParsePayload(packet.payload);
    const candidateValue =
      typeof parsed.value !== 'undefined'
        ? parsed.value
        : parsed.temperature ?? parsed.ph ?? parsed.stirring ?? parsed.raw;
    const numericValue = typeof candidateValue === 'number'
      ? candidateValue
      : Number(candidateValue);
    if (!Number.isNaN(numericValue)) {
      pushSample(telemetryKey, numericValue, parsed.ts ?? parsed.timestamp ?? Date.now());
      publishAggregateSnapshot();
    }
  }
});

// Publish retained setpoint state on startup.
publishSetpointState();
