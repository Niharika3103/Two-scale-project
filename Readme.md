1) After installing the required modules a minor change should be done within on module if the blind-watermark module is of version 0.4.2


2) goto venv-->lib-->site-packages--->blind-watermark--->blind-watermark.py


3) In the above mentioned file replace line no. 105 with below line of code.


4) byte = ''.join((np.round(wm)).astype(np.int_).astype(np.str_))

