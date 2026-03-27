import sounddevice as sd
import numpy as np
import time

# list all input devices
devices = sd.query_devices()
input_devices = []
for i, d in enumerate(devices):
    if d['max_input_channels'] > 0:
        input_devices.append((i, d))

print("Available input devices:")
for idx, (i, d) in enumerate(input_devices):
    print(f"  [{idx}] {d['name']} (channels: {d['max_input_channels']}, rate: {int(d['default_samplerate'])})")

print()
choice = int(input("Pick a device number: "))
dev_idx, dev = input_devices[choice]
native_rate = int(dev['default_samplerate'])

print(f"\nUsing: {dev['name']} at {native_rate} Hz")
print("Speak, clap, or tap your desk. Ctrl+C to quit.\n")

current_energy = 0.0

def callback(indata, frames, time_info, status):
    global current_energy
    if status:
        print(f"  !! {status}")
    current_energy = np.sqrt(np.mean(indata[:, 0] ** 2))

try:
    with sd.InputStream(samplerate=native_rate, channels=1, dtype='float32',
                        blocksize=int(native_rate * 0.1), callback=callback,
                        device=dev_idx):
        while True:
            e = current_energy
            bar_len = int(min(e * 300, 60))
            bar = '#' * bar_len + ' ' * (60 - bar_len)
            print(f"\r  |{bar}| {e:.6f}", end="", flush=True)
            time.sleep(0.05)
except KeyboardInterrupt:
    print("\ndone")
