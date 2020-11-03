import pandas as pd
import numpy as np

a = pd.Series()
a['1'] = 3
a['2'] = 3

b = a[a == 3].index
print(b)
print(b[1])