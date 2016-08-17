#showAntenna.py

* one_signal()
* two_signal()
* tmp()
* show_diff()

## one_signal()
#### Formula: (raw_siganl-background)/max(raw_siganl-background)
1. Each tx-rx signal is substracted the background signal, which is "basefile".
2. Absolute the output
3. Get the max number of each tx-rx signal.
4. Scale down the raw signal to [0-1]. 

## two_signal()
#### Use the same formula on two different raw signal. This function want to compare between two different targets with same pair raw signal.


## tmp()


## show_diff()


