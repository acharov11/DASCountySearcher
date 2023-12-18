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
This dataset contains the columns: county, state, lat, lon, total_population, population_density_per_sqmi, percent_fair_or_poor_health, percent_smokers, percent_adults_with_obesity, percent_physically_inactive, percent_excessive_drinking, teen_birth_rate, percent_uninsured, high_school_graduation_rate, percent_some_college, labor_force, percent_children_in_poverty, social_association_rate, violent_crime_rate, percent_severe_housing_problems, overcrowding, inadequate_facilities, life_expectancy, percent_frequent_physical_distress, percent_frequent_mental_distress, percent_adults_with_diabetes, percent_food_insecure, percent_insufficient_sleep, average_grade_performance, median_household_income, percent_enrolled_in_free_or_reduced_lunch, average_traffic_volume_per_meter_of_major_roadways, percent_homeowners, percent_severe_housing_cost_burden, percent_less_than_18_years_of_age, percent_65_and_over, percent_not_proficient_in_english, percent_female, percent_rural, per_capita_income, percentile_rank_below_poverty, percentile_rank_unemployed, percentile_rank_per_capita_income, percentile_rank_no_highschool_diploma, percentile_rank_socioeconomic_theme, percentile_rank_age_65_and_older, percentile_rank_age_17_and_younger, percentile_rank_disabled, percentile_rank_single_parent_households, percentile_rank_household_comp_disability_theme, percentile_rank_minorities, percentile_rank_limited_english_abilities, percentile_rank_minority_status_and_language_theme, percentile_rank_multi_unit_housing, percentile_rank_mobile_homes, percentile_rank_overcrowding, percentile_rank_no_vehicle, percentile_rank_institutionalized_in_group_quarters, percentile_rank_housing_and_transportation, percentile_rank_social_vulnerability, mean_winter_temp, mean_summer_temp.

The above list of variables are the variables we use in our analysis. The other variables in the original dataset were found to be insignificant for our purposes.
