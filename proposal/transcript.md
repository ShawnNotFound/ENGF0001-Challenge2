Now let’s talk about data analysis — this section is mainly software-based.

The collected sensor data — including temperature, pH, and stirring speed — are transmitted to the cloud, where an algorithm, as shown in the flowchart, continuously runs to analyze the data and send feedback if any modification is needed.

To understand the logic in more detail: once the system receives the current readings, it compares them with the target values and calculates the deviation for each parameter.

The model then uses two evaluation bands to decide whether the data represents an anomaly. The stop band is tighter and used when the previous data was already flagged as anomalous, while the start band is wider and applied when the system was previously normal.

If the deviation exceeds the corresponding band three consecutive times, the system confirms it as a real anomaly. The algorithm then sends correction instructions to the bioreactor based on the deviations.

This approach makes the control smarter and more stable, ensuring the bioreactor only reacts to persistent deviations rather than random sensor noise.

