import numpy as np
class DecisionTreeRegressor():
   # called every time an object is created from a class
    def __init__(self, min_samples_split=2, max_depth=2):
        ''' constructor '''
       
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split #specifies the minimum number of samples required to split an internal node
        self.max_depth = max_depth  #determines the maximum depth of the decision tree that will be constructed
    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''
        
        # dictionary to store the best split
        best_split = {}
        max_var_red = -float("inf")
        # loop over all the features in the dataset
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            #it will assighn the unique values in the dataset
            possible_thresholds = np.unique(feature_values)
            # loop over all the unique feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null 
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute varience reduction for target variable
                    curr_var_red = self.variance_reduction(y, left_y, right_y)
                    # update the best split if needed
                    # if curr_var_red is greater than max_var_red then it will update the best split as this value
                    if curr_var_red>max_var_red:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["var_red"] = curr_var_red
                        max_var_red = curr_var_red
                        
        # return best split
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        # if feature value or index is less than or equal to the threshold then the value is assighned to left 
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        # if feature value or index is greater than  to the threshold then the value is assighned to right 
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        # after splitiing it will return the result
        return dataset_left, dataset_right
    
    def variance_reduction(self, parent, l_child, r_child):
        ''' function to compute variance reduction gfeature'''
        
        weight_l = len(l_child) / len(parent)# it will calculate the varience of the left chaild
        weight_r = len(r_child) / len(parent)# it will calculate the varience of the right chld
        #after calculating varience of left and right child then using this we are going to calculate the varience reduction using this formula 
        #taking sum of the right and left child and subtracting with the varience of parent 
        reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
        #return the varience reduction
        return reduction
    
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        #it is used to find value of leaf node
        val = np.mean(Y)
        return val  

    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree '''
        #it will separate the data into independent and dependet 
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        #this dictionory will store best split value
        best_split = {}
        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["var_red"]>0:
              #if the best split varience reduction is greater than 0 it will build left and right subtree and increases the size of the depth +1
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["var_red"])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    
    #it will shows the how the tree will be build 
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.var_red)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y):
        ''' function to train the tree '''
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
        
    def make_prediction(self, x, tree):
        ''' function to predict new dataset '''
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        #the feature value is less than or equal to the threshol value of the tree it will make prediction on left tree else make prediction on right tree
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
    



    