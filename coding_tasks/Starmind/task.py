#!/usr/bin/env python
# coding: utf-8

# # Starmind Programming Task
# - First Solution using Levenshtein library
# - Second Solution using own implementation
# Code is the same as in ipynb, but for easy readabilty converted to py file


# In[1]:

""" How I would solve it: """
import csv

# wrapper only because I used it a second time below
def find_similar_names_in_csv(file_name, name, distance_function, distance):
  """ Finds similar names in a csv file through a distance function 
  Expecting the distance function to return integers"""
  names_arr = []
  with open(file_name, newline='') as csvfile:
    all_names = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in all_names:
      if len(row) > 0:
        # Depends if the names in brackets should count or not
        raw_name = row[0].split("(")[0].replace('"','').strip()
        if distance_function(raw_name, name) == distance:
          names_arr.append(raw_name)
  return set(names_arr)


# In[2]:


from Levenshtein import distance as levenshtein_distance
print("Results from Levenshtein Distance library:")
find_similar_names_in_csv("hundenamen.csv", "Luca", levenshtein_distance, 1)


# In[3]:


""" My own implementation of levenshtein distance from the top of my head (just because it's fun): """
import numpy as np

def update_2x2_matrix(arr, char1, char2):
  """ Updates the last value in the matrix with 3 values already calculated"""
  if char1 == char2:
    arr[1,1] = np.amin(arr)
  else:
    arr[1,1] = np.amin(arr) + 1
  
def my_levenshtein_distance(word1, word2):
  """ Calculates the levenshtein distance through a step by step matrix calculation (Not optimized)"""
  # Create initial matrix
  l_matrix = np.ones([len(word1)+1, len(word2)+1]) * np.inf
  l_matrix[:,0] = np.arange(l_matrix.shape[0])
  l_matrix[0,:] = np.arange(l_matrix.shape[1])
  
  # Iterate through matrix with 2x2 convolution-ish operation
  for i in range(l_matrix.shape[0]-1):
    for j in range(l_matrix.shape[1]-1):
      update_2x2_matrix(l_matrix[i:i+2, j:j+2], word1[i], word2[j])
  
  return int(l_matrix[-1,-1])


# In[4]:


print("Results from own Levenshtein Distance implementation:")
find_similar_names_in_csv("hundenamen.csv", "Luca", my_levenshtein_distance, 1)

