"""
Bayes Therorem:
    P(A | B) = P(B | A) * P(A) / P(B)


The following is just something that reminds me of what Bays Theorem  actually
is:
    Machine 1: 30 wrenches / hr           >>> P(Mach1) = 30 / 50 = 0.6
    Machine 2: 20 wrenches / hr           >>> P(Mach1) = 20 / 50 = 0.4

    Out of all produced parts:
    1% of wrenches are defective          >>> P(Defect) = 1%

    Out of all defective parts:
    50% came from machine 1               >>> P(Mach1 | Defect) = 50%
    50% came from machine 2               >>> P(Mach2 | Defect) = 50%

    Question:
    What is the probability that a part produced by machine 2 is defective?


"""

# Author: Yang Dai <daiy@mit.edu>


