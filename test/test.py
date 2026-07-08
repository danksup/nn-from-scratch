import numpy as np

corpus = "The first ten groups are families. So little is or was known about the last three groups that the author of the article classed together what are now known to be vast agglomerations of families. For instance, the American languages include several hundred distinct stocks, of which fifty are found in California alone. These are all, according to our present knowledge, utterly unrelated. It is known that the central African tongues belong to a different group than the southern, and it would be advisable to consult Sir Harry Johnston’s recent large work on the Bantu languages."
a = corpus.split()
b = np.lib.stride_tricks.sliding_window_view(a,2)
print(b[:,:-1])
print(b[:,1:])
print(b)