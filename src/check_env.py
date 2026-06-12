import sys, platform
import numpy as np
import pandas as pd
import scipy
import lightfm
from lightfm import LightFM
from scipy.sparse import random as sparse_random

print(f"Python: {sys.version.split()[0]}")
print(f"Platform: {platform.machine()}")
print(f"numpy: {np.__version__}")
print(f"pandas: {pd.__version__}")
print(f"scipy: {scipy.__version__}")
print(f"LightFM: {lightfm.__version__}")

# Тест многопоточности
m = sparse_random(1000, 500, density=0.05, format="csr", dtype=np.float32)
model = LightFM(loss="warp")
model.fit(m, epochs=2, num_threads=8)
print("✅ LightFM работает, многопоточность OK")