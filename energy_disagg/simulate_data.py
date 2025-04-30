import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)  # reproducibility

# Simulate 1 day at 1-minute intervals
minutes = 24 * 60
time = pd.date_range('2025-04-29', periods=minutes, freq='T')

# Base load: 300W constant (lights, chargers, etc.)
base_load = np.full(minutes, 300)

# Fridge: 150W cycles (5 min ON, 15 min OFF)
fridge = np.zeros(minutes)
for i in range(0, minutes, 20):
    fridge[i:i+5] = 150

# AC: 2000W from 14:00 to 18:00
ac = np.zeros(minutes)
ac[14*60:18*60] = 2000

# Microwave: 1200W random 5-min slots
microwave = np.zeros(minutes)
for _ in range(5):  # 5 random uses
    start = np.random.randint(8*60, 22*60)  # between 8 AM and 10 PM
    microwave[start:start+5] = 1200

# Total household load
household = base_load + fridge + ac + microwave

# Make DataFrame
df = pd.DataFrame({
    'timestamp': time,
    'power': household
})

df.set_index('timestamp', inplace=True)

# Plot a small section
df['power'].plot(figsize=(15,5))
plt.title('Simulated Household Energy Usage (1-min intervals)')
plt.ylabel('Watts')
plt.show()

df['delta'] = df['power'].diff().fillna(0)

# Set threshold for event detection (Watts)
threshold = 100  # only detect big jumps

# Find significant events
events = df[(df['delta'].abs() > threshold)]

print(events[['power', 'delta']].head(500))

def classify_event(delta):
    delta = abs(delta)
    if 1000 < delta < 1300:
        return 'Microwave'
    elif 140 < delta < 160:
        return 'Fridge'
    elif 1900 < delta < 2100:
        return 'AC'
    else:
        return 'Unknown'

# Apply classification
events['appliance'] = events['delta'].apply(classify_event)

print(events[['power', 'delta', 'appliance']].head(10))

plt.figure(figsize=(15,5))
plt.plot(df.index, df['power'], label='Household Power', alpha=0.5)
plt.scatter(events.index, events['power'], color='red', label='Detected Events', marker='x')

for idx, row in events.iterrows():
    plt.text(idx, row['power']+100, row['appliance'], fontsize=8, rotation=90)

plt.title('Detected Appliance Events')
plt.ylabel('Watts')
plt.legend()
plt.show()


