import os, sys
import time

print('>>>>>> ARIMA_MA print argv')
# time.sleep(30)

for arg in sys.argv:
    print('Arg Value = ', arg)

print('>>>>>> ARIMA_MA print env variables')

for a in os.environ:
    print(a, '=', os.getenv(a))

print('>>>>>> ARIMA_MA done')