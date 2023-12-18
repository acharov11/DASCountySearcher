import random

import folium
import os

color_list = []

for i in range(42):
    color = "#" + ''.join([random.choice('ABCDEF0123456789') for i in range(6)])
    color_list.append(color)

def map_plot_user(df, top_5_df = None):
    # Create a folium map centered around the mean latitude and longitude of the selected state
    map_center = [df['lat'].mean(), df['lon'].mean()]
    mymap = folium.Map(location=map_center, zoom_start=6)

    # Add markers to the map with tooltips only for the selected cluster
    for index, row in df.iterrows():
        #color = "#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])
        if (not top_5_df is None) and index in top_5_df.index:
            print(row['county'])
            folium.CircleMarker([row['lat'], row['lon']], popup=row['county'], color = 'red', fill_color = 'red',radius=5).add_to(mymap)
        else:
            folium.CircleMarker([row['lat'], row['lon']], popup=row['county'],color = 'blue', fill_color = 'blue',radius=2).add_to(mymap)

    # Specify the folder path where you want to save the HTML file
    output_folder = os.getcwd()
    selected_cluster = "cluster"
    # Save the interactive map as an HTML file in the specified folder
    html_file_path = output_folder + f'\\templates\\cluster_map_1_cluster_{selected_cluster}.html'
    # html_file_path = output_folder + f'\\templates\\cluster_map_{desired_states[0]}_cluster_{selected_cluster}.html'
    mymap.save(html_file_path)


def map_plot_all(df):
    # Create a folium map centered around the mean latitude and longitude of the selected state
    map_center = [df['lat'].mean(), df['lon'].mean()]
    mymap = folium.Map(location=map_center, zoom_start=6)

    # Add markers to the map with tooltips only for the selected cluster
    for index, row in df.iterrows():
        color = get_color_by_cluster(row['cluster'])
        folium.CircleMarker([row['lat'], row['lon']], popup=f"{row['county']} {row['cluster']}", color=color, fill_color=color,
                                radius=2).add_to(mymap)

    # Specify the folder path where you want to save the HTML file
    output_folder = os.getcwd()
    selected_cluster = "cluster"
    # Save the interactive map as an HTML file in the specified folder
    html_file_path = output_folder + f'\\templates\\cluster_map_1_cluster_all.html'
    # html_file_path = output_folder + f'\\templates\\cluster_map_{desired_states[0]}_cluster_{selected_cluster}.html'
    mymap.save(html_file_path)

def get_color_by_cluster(cluster):
    return color_list[cluster]
