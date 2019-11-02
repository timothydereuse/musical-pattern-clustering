# musical-pattern-clustering
Discovering patterns in monophonic music by clustering and distance learning.

See:
de Reuse, Timothy and Ichiro Fujinaga. "Pattern Clustering in Monophonic Music by Learning a Non-linear Embedding from Human Annotations." In _Proceedings of the 20th International Society for Music Information Retrieval Conference_. Delft, Netherlands, 2019.

### To Run:

Requires Python >= 3.7, pytorch >= 10.1, scikit-learn >= 0.21, music21, MATLAB.

Requires the PattDisc library (https://tomcollinsresearch.net/research/) to parse patterns using SIARCT-C.

Download the Meertens Tune Collection Annotated Corpus: (http://www.liederenbank.nl/mtc/) (Version 1.1)

`MTC2PointSet.py` will parse the .krn files from MTC-ANN and output them in a point-set format.

`siarct-c-test.m` will parse this point-set data and find patterns in each tune family.

`parseAndSavePatterns.py` will parse the output of `siarct-c-test.m`, extract features, and save as a big  `.pik` file. (This can take upwards of 15 minutes.)

`clusteringCrossValidation.py` will parse the saved `.pik` file, train a model to learn a distance metric based on the human annotations in MTC-ANN, and evaluate the resulting clustering based on ground truth (with default parameters, using 5-fold cross-validation).




