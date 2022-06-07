import math

# These are suggested helper functions
# You can structure your code differently, but if you have
# trouble getting started, this might be a good starting point

# returns whether a value is numeric or not
def isNumeric(value):
    return isinstance(value, int) or isinstance(value, float)


# partition dataset
def partition(xs, ys, col):

    # our partitioned data will be stored as a dictionary
    split = dict()

    # for each row in xs
    for i, row in enumerate(xs):

        # if the current row - col value is in split
        # append the row and each class to split
        if row[col] in split:
            split[row[col]][0].append(row)
            split[row[col]][1].append(ys[i])
        #  otherwise, add the values to split
        else:
            split[row[col]] = [row], [ys[i]]

    return split

# Calculate the entropy of a set of values 
# First count how often each value shows up 
# When you divide this value by the total number 
# of elements, you get the probability for that element 
# The entropy is the negation of the sum of p*log2(p) 
# for all these probabilities.
def entropy(values):

    # store the number of values we have
    numValues = counts(values)

    # store each entropy in values in a list
    entropies = list()

    for i in values:
        entropies.append((-numValues[i] / len(values)) * math.log(numValues[i] / len(values)))

    return sum(entropies)



# treeDeothCounter will keep track of the depth of the tree
treeDepthCounter = 0

# Create the decision tree recursively
def make_node(previous_ys, xs, ys, columns, maxDepth):
    # WARNING: lists are passed by reference in python
    # If you are planning to remove items, it's better 
    # to create a copy first
    columns = columns[:]

    global treeDepthCounter

    # First, check the three termination criteria:
    
    # If there are no rows (xs and ys are empty): 
    #      Return a node that classifies as the majority class of the parent
    if len(xs) == 0 and len(ys) == 0:
        return {"type": "class", "class": majority(previous_ys)}

    # If all ys are the same:
    #      Return a node that classifies as that class 
    if all(i == ys[0] for i in ys):
        return {"type": "class", "class": ys[0]}

    # If there are no more columns left:
    #      Return a node that classifies as the majority class of the ys

    #  Pre-pruning: OR if counter > 3 (meaning depth of the tree > 3)
    if len(columns) == 0 or treeDepthCounter > maxDepth:
        return {"type": "class", "class": majority(ys)}


    # Otherwise:
    # Compute the entropy of the current ys 
    currentYsEntropy = entropy(ys)
    
    # track the most gain from a split and which split
    # resulted in the most gain
    mostGain = -1000
    mostGainPartition = dict()

    # tracks indexes and columns that will be removed
    indexToRemove = 0
    columnToRemove = 0

    # For each column:
    for i, col in enumerate(columns):

        # Perform a split on the values in that column 
        partitionedData = partition(xs, ys, col)

        # if the partitioned data is too small, return a decision node 
        # based on the majority of the current classes
        # (pre-pruning)
        if(len(partitionedData) < 2):
            return {"type": "class", "class": majority(ys)}


        # Calculate the entropy of each of the pieces
        # Compute the overall entropy as the weighted sum 
        entropiesOfThisSplit = list()

        # look for smallest entropy to find gain
        # minEntrpoy = 1000

        for j in partitionedData:

            # get the labels from each element in partitioned data
            partitoinedLabels = partitionedData[j][1]

            # add the current entropy to the list
            entropiesOfThisSplit.append(len(partitoinedLabels) / len(ys) * entropy(partitoinedLabels))
        
        # get the sum of the entropies from this split
        splitEntropy = sum(entropiesOfThisSplit)
    
        # The gain of the column is the difference of the entropy before
        # the split, and this new overall entropy 
        infoGain = currentYsEntropy - splitEntropy

        """
        if currentEntropy < minEntropy:
            minEntropy = currentEntropy
        """

        # if the current gain is greater than the current highest gain
        if infoGain > mostGain:

            # replace the highest gain
            mostGain = infoGain
            mostGainSplit = partitionedData

            # store the index that will not be included
            indexToRemove = i

    # remove the appropriate column
    columnToRemove = columns[indexToRemove]
    del columns[indexToRemove]

    # Select the column with the highest gain, then:
    # Split the data along the column values and recursively call 
    # make_node for each piece 

    # Create a split-node that splits on this column, and has the result 
    #    of the recursive calls as children.
    children = dict()
    for i in mostGainSplit:
        children[i] = make_node(ys, mostGainSplit[i][0], mostGainSplit[i][1], columns, maxDepth)

    # increment counter
    treeDepthCounter += 1

    # return node
    return {"type": "split", "split": columnToRemove, 'children': children}

    
    

# Determine if all values in a list are the same 
# Useful for the second basecase above
def same(values):
    
    # if values is not empty
    if not values.isEmpty():

        # pick the first item
        firstItem = values[0]

        # loop through the rest of the list to see that they are all the same
        for i in range(1, len(values)):
            if values[i] != firstItem:
                return False
        
    return True


    
# Determine how often each value shows up 
# in a list; this is useful for the entropy
# but also to determine which values is the 
# most common
def counts(values):

    countsDict = {}

    # loop through values and update dictionary counts
    for i in values:
        if i in countsDict:
            countsDict[i] += 1
        else:
            countsDict[i] = 1

    return countsDict
   

# Return the most common value from a list 
# Useful for base cases 1 and 3 above
def majority(values):
    valCounts = counts(values)

    # initially minimize max variable
    max = -1000
    maxValue = 0

    # for each value
    for i in valCounts:

        #  if the current  value > the max value
        if valCounts[i] > max:

            # replace max value
            max = valCounts[i]
            maxValue = i

    return maxValue
    
    


# This is the main decision tree class 
# DO NOT CHANGE THE FOLLOWING LINE
class DecisionTree:
# DO NOT CHANGE THE PRECEDING LINE

    counter = 0
    def __init__(self, tree={}):
        self.tree = tree
    
    # DO NOT CHANGE THE FOLLOWING LINE    
    def fit(self, x, y):
    # DO NOT CHANGE THE PRECEDING LINE
        global treeDepthCounter
        treeDepthCounter = 0
        self.majority = majority(y)
        self.tree = make_node(y, x, y, list(range(len(x[0]))), 3)


    # To classify using the tree:
    def classify(self, currentNode, x):

        # As long as the current node is an interior node (type == "split"):
        if currentNode["type"] == "split":
            #    get the value of the attribute the split is performed on 
            #    select the child corresponding to that value as the new current node 
            splitVal = x[currentNode["split"]]

            # NOTE: In some cases, your tree may not have a child for a particular value 
            #       In that case, return the majority value (self.majority) from the training set 
            if splitVal not in currentNode["children"]:
                return self.majority

            return self.classify(currentNode["children"][splitVal], x)
        else:
            return currentNode["class"]
        
    
    # DO NOT CHANGE THE FOLLOWING LINE    
    def predict(self, x):
    # DO NOT CHANGE THE PRECEDING LINE    
        if not self.tree:
            return None
        
        # IMPORTANT: You have to perform this classification *for each* element in x 
        # Note that the result is a list of predictions, one for each x-value
        return list(self.classify(self.tree, i) for i in x)
    
    # DO NOT CHANGE THE FOLLOWING LINE
    def to_dict(self):
    # DO NOT CHANGE THE PRECEDING LINE
        # change this if you store the tree in a different format
        return self.tree
