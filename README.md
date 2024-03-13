# DASCountySearcher
## Team DAS - Danny Silverstein, Anton Charov, and Sam Barbeau

# Overview: 
Our U.S. County Recommender System is a user-friendly application designed to assist users in finding their most suitable counties in the United States to live in based on their personal preferences and demographic information. By collecting a variety of data points ranging from age, race, and preferred temperature to additional specific preferences, our system provides tailored recommendations aiming to enhance the user’s quality of life and overall satisfaction in their new locale.

# Languages: 
Python (cleaning and implementing), R (cleaning and analyzing), SQL (analyzing)

# Goal:
By prompting the user with questions, we will generate a temporary user "county". Using this data, we will assign the user to a group of similar counties after a series of fundemental statistical and ML techniques (i.e., PCA and k-means). Given our imense number of columns as seen below, our questions to the user will be prompted to answer only the most important variables in our data; the remaining variables are imputed.

After matching users to their ideal counties after thorough analysis and series of questions, the counties will be displayed on an interactive map using the Folium library.

# Citations/Datasets:
John Davis (johnjdavisiv), Kaggle: US Counties: COVID19 + Weather + Socio/Health data (big_dataset.csv). 
This dataset contains the columns: county, state, lat, lon, etc..
