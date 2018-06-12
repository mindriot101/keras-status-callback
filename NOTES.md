# TODO

- Include confusion matrix at end of each epoch
    - can be implemented with callback, but must be run on all the test
      data each time
- Switch back to SGD

## Items to track for each run

- model persisted to disk
- files used for training and validation
    - training good
    - training defect
    - validation good
    - validation defect
    - test good
    - test defect
- serialized model (if possible)
- optimiser settings?
- grayscale yes/no
- undersampling yes/no

## Items to track at the end of each epoch

- confusion matrix (we can derive all accuracy-based stats from this)
- timestamp of each epoch
