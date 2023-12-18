import os
import random
import pandas as pd
import folium
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import utils


# self - note: have option to answer all 58 questions, and maybe recommend answering
# the 10 selected for a quick evaluation
# to get good evaluation you should try aLl 58

np.random.seed = 42

# Grab data
data = pd.read_csv("C:\\data\\school\\DataScience\\CSVs\\county_data_PCA.csv")
#data = data[data['state'] == 'Illinois']

only_states = str(input("Do you want state-specific search? (Y/n): ")).lower()
only_state = False
if only_states == 'y':
    only_state = True

if only_state:
    desired_states_str = input("Enter the states you want to see, separated by commas (e.g., 'Illinois,California'): ").capitalize()
    desired_states = [state.strip() for state in desired_states_str.split(',')]
    data = data[data['state'].isin(desired_states)].copy()

# Remove categorical variables
#data = data.drop(['percentile_rank_age_65_and_older','percentile_rank_age_17_and_younger','percentile_rank_overcrowding'],axis = 1)
#data = data.drop(['percent_65_and_over','percent_less_than_18_years_of_age','overcrowding'],axis = 1)
numerical_data = data.drop(['county', 'state', 'lat', 'lon'], axis=1)

# Standardize the data
scaler = StandardScaler()
scaler.fit(numerical_data)
scaled_data = scaler.transform(numerical_data)

# Perform PCA -- store to numpy array
NUM_COMPONENT = 10
# for state -- handle small states like Hawaii
if only_state:
    if len(data) < 10 :
        NUM_COMPONENT = len(data)
###
pca = PCA(n_components=NUM_COMPONENT)
pca.fit(scaled_data)
np_pca = pca.transform(scaled_data)

# Get loadings (weights) of each principal component
loadings = pca.components_

# Calculate the cumulative contribution of each feature
cumulative_contributions = np.sum(np.abs(loadings), axis=0)

# Identify the indices of the top # features
NUM_TOP_FEATURES = 15
top_feature_indices = np.argsort(cumulative_contributions)[-NUM_TOP_FEATURES:]

# Get the names of the top # features
feature_names = numerical_data.columns[top_feature_indices]

print(f"Top {len(feature_names)} features based on cumulative contributions across all principal components:")
print(feature_names)


# elbow_method(np_pca)
# pass in np_pca
# WARNING computationally expensive!
def elbow_method(pca_data):
    sse = {}
    for k in range(1, 30):
        kmeans = KMeans(n_clusters=k, max_iter=1000, n_init=10).fit(pca_data)
        data["clusters"] = kmeans.labels_
        # print(data["clusters"])
        sse[k] = kmeans.inertia_  # Inertia: Sum of distances of samples to their closest cluster center
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.show()

# based on elbow method
NUM_CLUSTERS = 42 # for US
 # for states
# small state check
if only_state:
    NUM_CLUSTERS = 5
    if len(data) < 5:
        NUM_CLUSTERS = len(data)
##
# Explicitly set value of n_init to suppress warning
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
data['cluster'] = kmeans.fit_predict(np_pca)


# EXAMPLE: plot all k-means clusters
# utils.map_plot_all(data)
# EXAMPLE: plot user selected counties with cluster 37
# utils.map_plot_user(data[data.cluster==37])
# utils.map_plot_user(data[data.cluster==23])
# EXAMPLE: query based on county/state
# data[data['county'] == 'Lake'] @ query lake
# data[data['state'] == 'Illinois']


# Takes in preference and scales it to the predictor/question, EX: population: (1,000-10,000), 5 -> 10,000, 3 -> 5000
# Note when answering questions: put in 3 when I didn't care, which is the median. Makes sense to impute with median.
# serves as middle base
def get_scaled_value(num_data, feature_name, user_question_response):
    feature_column = num_data[feature_name]
    if user_question_response == 1:
        return min(feature_column)
    elif user_question_response == 2:
        return feature_column.quantile(0.25)
    elif user_question_response == 3:
        return feature_column.median()
    elif user_question_response == 4:
        return feature_column.quantile(0.75)
    elif user_question_response == 5:
        return max(feature_column)


# Runs the scalar to predictor function and updates all user inputs to this value
# EX: set_input_from_ans([3,2,2,4,4,4,1,4,2,3], some_user_factor_inputs)
# User answers: [4.0, 1.0, 4.0, 2.0, 5.0, 3.0, 5.0, 4.0, 3.0, 5.0, 4.0, 2.0, 4.0, 3.0, 3.0] this gives cook county
def set_input_from_ans(user_ans, user_factor_input):
    for i in range(len(user_ans)):
        val = get_scaled_value(numerical_data, feature_names[i], user_ans[i])
        user_factor_input[feature_names[i]] = val


# Map predictors to corresponding questions
factor_to_question_map = {
    'total_population': "How large would you like the county population to be? (1 - Smallest, 5 - Largest)",
    'population_density_per_sqmi': "How crowded do you want the county to be? (1 - Least, 5 - Most)",
    'percent_fair_or_poor_health': "How much does the overall health of the community influence your decision? (1 - Not at all, 5 - Very much)",
    'percent_smokers': "How concerned are you about smoking rates in the area? (1 - Very concerned, 5 - Not at all concerned)",
    'percent_adults_with_obesity': "How concerned are you about the lack of quality food? (1 - Very concerned, 5 - Not concerned)",
    'percent_physically_inactive': "How concerned are you about the level of physical inactivity in the area (running, walking, biking)? (1 - Very concerned, 5 - Not at all concerned)",
    'percent_excessive_drinking': "How concerned are you about alcohol overconsumption in your county? (1 - Very concerned, 5 - Not at all concerned)",
    'teen_birth_rate': "How much does the teen birth rate concern you? (1 - Very concerned, 5 - Not at all concerned)",
    'percent_uninsured': "How concerned are you about the number of uninsured people in your area? (1 - Very concerned, 5 - Not at all concerned)",
    'high_school_graduation_rate': "How important is the high school quality in the area to you? (1 - Not important, 5 - Very important)",
    'percent_some_college': "How much do you value college education? (1 - Not important, 5 - Very important)",
    'labor_force': "How important is the size of the labor force to you? (1 - Not important, 5 - Very important)",
    'percent_children_in_poverty': "How concerned are you about child poverty in the area? (1 - Very concerned, 5 - Not at all concerned)",
    'social_association_rate': "How important is a strong sense of community to you? (1 - Not important, 5 - Very important)",
    'violent_crime_rate': "How concerned are you about the level of violent crime in the area (Car jacking, assault, robbery, etc)? (1 - Very concerned, 5 - Not at all concerned)",
    'percent_severe_housing_problems': "How concerned are you about housing damage in the area (damaged roofs, roads, etc)? (1 - Very concerned, 5 - Not at all concerned)",
    'overcrowding': "How much does it bother you if some homes have too many people living in them, more than what's safe? (1 - Really bothers me, 5 - Doesn't bother me at all)", # ???
    'inadequate_facilities': "How concerned are you about households lacking resources like hot water, washing machines, and air conditioners? (1 - Very concerned, 5 - Not at all concerned)",
    'life_expectancy': "How important is life expectancy in your decision-making process? (1 - Not important, 5 - Very important)",
    'percent_frequent_physical_distress': "To what extent does the frequency of physical injury in the area concern you? (1 - Very concerned, 5 - Not at all concerned)",
    'percent_frequent_mental_distress': "How concerned are you about living in a stressful environment? (1 - Very concerned, 5 - Not at all concerned)",
    'percent_adults_with_diabetes': "How concerned are you about the percentage of adults with chronic illnesses? (1 - Very concerned, 5 - Not at all concerned)",
    'percent_food_insecure': "How important is the availability of food (supermarkets, grocery stores)? (1 - Not important, 5 - Very important)",
    'percent_insufficient_sleep': "How concerned are you about not getting enough sleep? (1 - Very concerned, 5 - Not concerned)",
    'average_grade_performance': "How much does the academic performance in the area matter to you? (1 - Not important, 5 - Very important)",
    'median_household_income': "How wealthy would you like your county population to be? (1 - Lowest income, 5 - Highest income)",
    'percent_enrolled_in_free_or_reduced_lunch': "How much does it bother you that some students get free or discounted lunch? (1 - Really bothers me, 5 - Doesn't bother me at all)",
    'average_traffic_volume_per_meter_of_major_roadways': "How concerned are you about the amount of traffic in your area? (1 - Very concerned, 5 - Not at all concerned)",
    'percent_homeowners': "How important is it that members of the community own their homes (not renting/leasing)? (1 - Not important, 5 - Very important)",
    'percent_severe_housing_cost_burden': "How worried are you about having a hard time paying for your home? (1 - Very worried, 5 - Not worried at all)",
    'percent_less_than_18_years_of_age': "How much do you care about the number of children in the area? (1 - Least kids, 5 - Most kids)",
    'percent_65_and_over': "How much do you care about the number of elderly in the area? (1 - Least elderly, 5 - Most elderly)",
    'percent_not_proficient_in_english': "How concerned are you about the percent of the population that doesn't speak English? (1 - Very concerned, 5 - Not at all concerned)",
    'percent_female': "What type of gender spread would you like? (1 - Mostly males, 3 - Even, 5 - Mostly females)",
    'percent_rural': "How rural would you like your area to be (lots of farming/open land)? (1 - Not rural, 5 - Very rural)",
    'per_capita_income': "How much does average income in the area matter to you? (1 - Not important, 5 - Very important)",
    'percentile_rank_below_poverty': "How concerned are you about poverty in the area? (1 - Very concerned, 5 - Not concerned)",
    'percentile_rank_unemployed': "How important is the number of unemploymed people in the area? (1 - Not important, 5 - Very important)",
    'percentile_rank_per_capita_income': "How important is it that your community has high paying jobs? (1 - Not important, 5 - Very important)",
    'percentile_rank_no_highschool_diploma': "How much do you value graduating high school? (1 - Absolutely necessary , 5 - Not required)",
    'percentile_rank_socioeconomic_theme': "How much do you care about everyone having similar amounts of money in the place you live? (1 - Don't care, 5 - Care a lot)",
    'percentile_rank_disabled': "How important is the number of individuals with disabilities to you? (1 - Not important, 5 - Very important)",
    'percentile_rank_single_parent_households': "How much do you care about the number of single-parent households? (1 - Don't care, 5 - Care a lot)",
    'percentile_rank_household_comp_disability_theme': "How much does the availability of handicap-friendly structure matter to you (ramps, elevators)? (1 - Not important, 5 - Very important)",
    'percentile_rank_minorities': "How important is it to live in a diverse community?  (1 - Not important, 5 - Very important)",
    'percentile_rank_limited_english_abilities': "How important is it that you live among english speakers?  (1 - Not important, 5 - Very important)",
    'percentile_rank_minority_status_and_language_theme': "How important is it that you live in an area with minorities? (1 - Not important, 5 - Very important)",
    'percentile_rank_multi_unit_housing': "How important is it that you live in an apartment? (1 - Not important, 5 - Very important)",
    'percentile_rank_mobile_homes': "How much would you like mobile homes in your area (trailer, RV, trailer home)? (1 - No mobile homes, 5 - Many mobile homes)",
    'percentile_rank_no_vehicle': "How important is public transportation to you? (1 - Not important, 5 - Very important)",
    'percentile_rank_institutionalized_in_group_quarters': "How concerned are you about the number of individuals in institutions (e.g. nursing homes, prisons, etc)? (1 - Very concerned, 5 - Not concerned)",
    'percentile_rank_housing_and_transportation': "How important is it that you live in a city-like setting? (1 - Not important, 5 - Very important)",
    'percentile_rank_social_vulnerability': "How concerned are you about being affected by external hazards (e.g. Natural disasters, Public health crises)? (1 - Very concerned, 5 - Not concerned)",
    'mean_winter_temp': "In the winter, how cold do you prefer it to be? (1 - Not cold, 5 - Very cold)",
    'mean_summer_temp': "In the summer, how hot do you prefer it to be? (1 - Not hot, 5 - Very hot)",
    'percentile_rank_age_65_and_older': "How important is it that you move to a common retirement setting? (1 - Not important, 5 - Very important)",
    'percentile_rank_age_17_and_younger': "How important is it that you move to an education-focused setting? (1 - Not important, 5 - Very important)",
    'percentile_rank_overcrowding': "How claustrophobic would you consider yourself? (1 - Not claustrophobic, 5 - Very claustrophobic)"
}
# dictionary to store scaled user responses, key: factor, value: value
user_factors = {}
# Fill with medians to impute base data -- answers they do not provide
for column in numerical_data.columns:
    user_factors[column] = numerical_data[column].median()
# Append user values
user_inputs = []
# Prompt the user to rate their preferences on a scale of 1-5 for each column
for feature in feature_names:
    question = factor_to_question_map[feature]
    preference = float(input(f"{question}: "))
    while preference < 1 or preference > 5:
        preference = float(input(question))
    user_inputs.append(preference)
# convert all user inputs to percentile values from factor values
set_input_from_ans(user_inputs, user_factors)
print("---------------")
print(f"User answers: {user_inputs}")
# print(f"User data: {user_factors.values()}")
print("---------------")

# Create a DataFrame for the user county
user_data = pd.DataFrame(user_factors, index=[0])

# Standardize the user data using the same scaling as data
standardized_user_data = scaler.transform(user_data)

# Perform PCA for the user county
pca_user = pca.transform(standardized_user_data)

# Predict the cluster for the user county
user_cluster = kmeans.predict(pca_user)[0]

# get all values in data from the predicted cluster
user_cluster_data = data[data['cluster'] == user_cluster]
# reindex the list, we have a bunch of indexes starting at 60, 89, this will make it start at 0 again
user_cluster_data = user_cluster_data.reset_index().drop(['index'], axis=1)
# work in the same transform space
user_cluster_data_scaled = scaler.transform(
    user_cluster_data.drop(['county', 'state', 'lat', 'lon', 'cluster'], axis=1))
# store user PCA loadings for distance function
user_cluster_data_loadings = pca.transform(user_cluster_data_scaled)


# Helpful analytics commands
# user_cluster_data_loadings.shape
# user_cluster_data['state'].unique()
# user_cluster_data[user_cluster_data['state'] == 'Illinois']
# pca.transform(user_cluster_data_scaled)[0]

# Find the closest points in multidimensional space -- we have 58 dimensions
# Calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


# Calculate Euclidean distances from each point in the user cluster loadings (matched by cluster to
# the original dataset and rescaled by PCA to reduce to 10 dimensions) to the user inputted pca loadings
# simple multidimensional distance good for finding most similar counties
distances = np.array([euclidean_distance(pca_user, point) for point in user_cluster_data_loadings])
user_cluster_data['distance'] = distances
# select the 5 closest counties
user_cluster_data_top_5 = user_cluster_data.sort_values('distance').head(5)
# plot and generate map
utils.map_plot_user(user_cluster_data, user_cluster_data_top_5)

