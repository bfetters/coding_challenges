## Fill each each function stub according to the docstring.
## Run the tests with this command: "make test"

import numpy as np
import pandas as pd
import requests
from collections import Counter,defaultdict
from itertools import izip, count
import requests
import json
import wikipedia

### Python
def count_characters(string):
    '''
    INPUT: STRING
    OUTPUT: DICT (STRING => INT)

    Return a dictionary which contains a count of the number of times each
    character appears in the string.
    Characters which would have a count of 0 should not need to be included in
    your dictionary.
    '''
    count = Counter(string)
    return count

def invert_dictionary(d):
    '''
    INPUT: DICT (STRING => INT)
    OUTPUT: DICT (INT => SET OF STRINGS)

    Given a dictionary d, return a new dictionary with d's values as keys and
    the value for a given key being the set of d's keys which have the same
    value.
    e.g. {'a': 2, 'b': 4, 'c': 2} => {2: {'a', 'c'}, 4: {'b'}}
    '''
    dInv = defaultdict(set)
    for k,v in d.iteritems():
        dInv[v].update(set(k))
    return dInv

def word_count(filename):
    '''
    INPUT: STRING
    OUTPUT: (INT, INT, INT)

    filename refers to a text file.
    Return a tuple containing these stats for the file in this order:
      1. number of lines
      2. number of words (broken by whitespace)
      3. number of characters
    '''
    lines = 0
    words = 0
    letters = 0
    with open(filename) as fh:
        keep_lines = fh.readlines()
        
    for line in keep_lines:
        lines += 1
        for word in line.split(' '):
            words += 1
            
        for letter in line:
            letters += 1
        
    return (lines, words, letters)

def matrix_multiplication(A, B):
    '''
    INPUT: LIST OF LIST OF INTEGERS, LIST OF LIST OF INTEGERS
    OUTPUT: LIST OF LIST of INTEGERS

    A and B are matrices with integer values, encoded as lists of lists:
    e.g. A = [[2, 3, 4], [6, 4, 2], [-1, 2, 0]] corresponds to the matrix:
    | 2  3  4 |
    | 6  4  2 |
    |-1  2  0 |
    Return the matrix which is the product of matrix A and matrix B.
    You may assume that A and B are square matrices of the same size.
    
    You may not use numpy. Write your solution in straight python.
    '''
    result = np.zeros((len(A),len(A))).tolist()
    # iterate through rows of A
    for i in range(len(A)):
        # iterate through columns of B
        for j in range(len(B[0])):
            # iterate through rows of B
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
                
    return result

def max_lists(list1, list2):
    '''
    INPUT: list, list
    OUTPUT: list

    list1 and list2 have the same length. Return a list which contains the
    maximum element of each list for every index.
    '''
    max_list = []
    for i in zip(list1, list2):
        max_list.append(max(i[0],i[1]))
    return max_list

def get_diagonal(mat):
    '''
    INPUT: 2 dimensional list
    OUTPUT: list

    Given a matrix encoded as a 2 dimensional python list, return a list
    containing all the values in the diagonal starting at the index 0, 0.

    E.g.
    mat = [[1, 2], [3, 4], [5, 6]]
    | 1  2 |
    | 3  4 |
    | 5  6 |
    get_diagonal(mat) => [1, 4]

    You may assume that the matrix is nonempty.
    '''
    diagonal = []
    for i in xrange(len(mat)):
        print i
        diagonal.append(mat[i][i])
    return diagonal

def merge_dictionaries(d1, d2):
    '''
    INPUT: dictionary, dictionary
    OUTPUT: dictionary

    Return a new dictionary which contains all the keys from d1 and d2 with
    their associated values. If a key is in both dictionaries, the value should
    be the sum of the two values.
    '''
    merged = d1.copy()
    for k in d2:
        if k in merged:
            merged[k] += d2[k]
        else:
            merged[k] = d2[k]
            
    return merged

def make_char_dict(filename):
    '''
    INPUT: string
    OUTPUT: dictionary (string => list)

    Given a file containing strings, you would like to create a dictionary with keys
    of single characters. The value is a list of all the line numbers which
    start with that letter. The first line should have line number 1.
    Characters which never are the first letter of a line do not need to be
    included in your dictionary.
    '''
    with open(filename) as fh:
        keep_lines = fh.readlines()
     
    my_dict = {}
    for i,line in enumerate(keep_lines):
        if line[0] in my_dict:
            my_dict[line[0]].append(i + 1)
        else:
            my_dict[line[0]] = [i + 1]
    return my_dict

### NumPy
def array_work(rows, cols, scalar, matrixA):
    '''
    INPUT: INT, INT, INT, NUMPY ARRAY
    OUTPUT: NUMPY ARRAY

    Create matrix of size (rows, cols) with the elements initialized to the
    scalar value. Right multiply that matrix with the passed matrixA (i.e. AB,
    not BA). 
    Return the result of the multiplication.
    You should be able to accomplish this in a single line.

    Ex: array_work(2, 3, 5, [[3, 4], [5, 6], [7, 8]])
           [[3, 4],      [[5, 5, 5],
            [5, 6],   *   [5, 5, 5]]
            [7, 8]]
    '''
    B = np.linspace(scalar,scalar, rows*cols).reshape(rows,cols)
    return np.dot(matrixA,B)

def boolean_indexing(arr, minimum):
    '''
    INPUT: NUMPY ARRAY, INT
    OUTPUT: NUMPY ARRAY

    Returns an array with all the elements of "arr" greater than
    or equal to "minimum"

    Ex:
    In [1]: boolean_indexing([[3, 4, 5], [6, 7, 8]], 7)
    Out[1]: array([7, 8])
    '''
    result = []
    for row in xrange(len(arr)):
        for column in xrange(len(arr[0])):
            if arr[row][column] >= minimum:
                result.append(arr[row][column])
    return result

### Pandas
def make_series(start, length, index):
    '''
    INPUT: INT, INT, LIST
    OUTPUT: PANDAS SERIES

    Create a pandas Series of length "length"; its elements should be
    sequential integers starting from "start". 
    The series' index should be "index". 

    Ex: 
    In [1]: make_series(5, 3, ['a', 'b', 'c'])
    Out[1]: 
    a    5
    b    6
    c    7
    dtype: int64
    '''
    lst = []
    for l in xrange(start,start+length):
        lst.append(l)
     
    ser = pd.Series(data=lst,index=index)
    return ser

def data_frame_work(df, colA, colB, colC):
    '''
    INPUT: DATAFRAME, STR, STR, STR
    OUTPUT: None
    
    Insert a column (colC) into the dataframe that is the sum of colA and colB.
    '''
    df[colC] = df[colA] + df[colB]

# For each of these, you will be dealing with a DataFrame which contains median
# rental prices in the US by neighborhood. The DataFrame will have these
# columns:
# Neighborhood, City, State, med_2011, med_2014
def pandas_add_increase_column(df):
    '''
    INPUT: DataFrame
    OUTPUT: None

    Add a column to the DataFrame called 'Increase' which contains the 
    amount that the median rent increased by from 2011 to 2014.
    '''
    df['Increase'] = df.med_2014 - df.med_2011

def pandas_only_given_state(df, state):
    '''
    INPUT: DataFrame, string, string
    OUTPUT: DataFrame

    Return a new pandas DataFrame which contains the entries for the given
    state. Only include these columns:
        Neighborhood, City, med_2011, med_2014
    '''
    df_state = df[df['State'] == state]
    df_state = df_state.drop('State', axis=1)
    return df_state

def pandas_max_rent(df):
    '''
    INPUT: DataFrame
    OUTPUT: DataFrame

    Return a new pandas DataFrame which contains every city and the highest
    median rent from that city for 2011 and 2014.

    Note that city names are not unique and you need to use the state as well
    so that Portland, ME and Portland, OR are recognized as different.

    Your DataFrame should contain these columns:
        City, State, med_2011, med_2014
    '''
    df['CityState'] = df.City + df.State
    df_11 = df.drop(['Neighborhood','med_2014'],axis=1).sort(['med_2011'],ascending=False)
    df_11_grp = df_11.groupby(['CityState']).first()
    df_14 = df.drop(['Neighborhood','med_2011','City','State'],axis=1).sort(['med_2014'],ascending=False)
    df_14_grp = df_14.groupby(['CityState']).first()
    df_merged = df_11_grp.merge(df_14_grp, left_index="CityState", right_index="CityState")
    df_merged.reset_index()
    return df_merged

# APIs
def api_parse(url):
    """
    INPUT: STR (URL)
    OUTPUT: INTEGER (Number of links in Wikipedia article)
    
    Create a function that given a URL of a particular Wikipedia article,
    returns the number of links in that article as seen by the API.
    """
    
    headers = {'user_agent': 'DataWrangling/1.1 (http://galvanize.it; class@galvanize.it)'}

    payload = 'action=parse'
    r = requests.post(url, data=payload, headers=headers)
    links = r.json()['parse']['links']
    return len(links)

def pymongo_count(collection):
    '''
    INPUT: pymongo collection object
    OUTPUT: INTEGER

    return the number of mongoDB documents
    that have EXACTLY than 1 'links'.
    
    example use to inspect the schema of one document:


    collection.find_one({})
    '''
    return collection.find({'links':{'$size':1}}).count()
