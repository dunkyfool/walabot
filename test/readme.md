# log5.py

* timeout() 
* init()
* convert_log2data()
* convert_label()
* main

1. Do not forget to change the MODE @line-14 (training:MODE='train', valadation:MODE='val', test:MODE='test')
2. Please uncomment from line 130 to line 156

## timeout() 
It will automatically skip when the user do no enter any keyword.

## init()
Default the specific or all dataset.

## convert_log2data()
Convert the raw signal into data.

## convert_label()
Convert the keyword into label.

## main
1. Manually key in the label.(current value is distance of the target)
2. Initial the dataset.
3. Overwrite the dataset or not
4. Show name and shape of each dataset.
5. Execute buildAll.sh and convert the value to dataset.
