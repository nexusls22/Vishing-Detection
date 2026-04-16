import os
import multiprocessing

print(f"Logische CPU-Kerne (os): {os.cpu_count()}")
print(f"Logische CPU-Kerne (multiprocessing): {multiprocessing.cpu_count()}")