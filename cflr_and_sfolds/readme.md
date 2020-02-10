Jeremy Herman

cflr.py:
- To run: `python3 cflr.py`
- By default this uses the 'x06Simple.csv', but you could run `python3 cflr.py "other.csv"`

sfolds.py:
- To run: `python3 sfolds.py n`
- requires a 3rd argument
- n should be the number of s-folds you want. Entering a number works, but also entering 'n' will make s = number of samples
- By default this uses the 'x06Simple.csv', but you could run `python3 cflr.py n "other.csv"`


There is a basic makefile included that will run a few tests by running `make run`.
Test output is included in the pdf and in the file 'test_outputs_1'
