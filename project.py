import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


#read the data from csv file
dataset = pd.read_csv('world_population_data.csv')

# Show all columns
pd.set_option('display.max_columns', None)

print("\n---------Print the first 5 rows of the DataFrame----------\n")
print(dataset.head(5))



print("\n--------Print the last 5 rows of the DataFrame--------\n")
print(dataset.tail(5))

print("\n index: ",dataset.index)

print("\n--------Information about the dataset--------\n")
print(dataset.info())

print("\n---------Summary Of The Dataset---------\n")
print(dataset.describe())

print("\n--------Null values in the entire dataset--------\n")
print(dataset.isnull().sum())

print("--------duplicate rows in the dataset---------\n")
print(" \nDuplicate raw : ", dataset.duplicated().sum())

# Select the years columns
years = [
     '2000 population',
     '2010 population',
     '2015 population',
     '2020 population',
     '2022 population',
     '2023 population']


#Finding mode and frequency table to find out which continents have the most countries
mode_continent = dataset['continent'].mode()[0]
print("\nMode of Continent:", mode_continent)

frequency_table_continent = dataset['continent'].value_counts()
print("\nFrequency table for 'continent' column:")
print(frequency_table_continent)


mean_population =[]
median_population = []

#Finding median and mean of all numerical columns

#finding mean and median of population columns
for x in years:
    mean_population.append(dataset[x].mean())
    median_population.append(dataset[x].median())

mean_area = dataset['area (km²)'].mean()
median_area = dataset['area (km²)'].median()

mean_density = dataset['density (km²)'].mean()
median_density = dataset['density (km²)'].median()
# mean_density = dataset['density_km2'].mean()
# median_density = dataset['density_km2'].median()

print("\nMean population-----------------")
for i in range(len(years)):
    print(years[i], " : ", mean_population[i])

print("\nMedian population---------------")
for i in range(len(years)):
    print(years[i], " : ", median_population[i])


print("\nMean Area (km²):", mean_area)
print("Median Area (km²):", median_area)


print("\nMean Density (km²):", mean_density)
print("Median Density (km²):", median_density)


data_range =dataset['density (km²)'].max() - dataset['density (km²)'].min()
print("\nRange of density column :", data_range)

data_variance = np.var(dataset['density (km²)'])
print("\nVariance of density column :", data_variance)

# Select the numerical columns only for finding the correlation
numerical_columns = dataset.select_dtypes(include=['number'])




# find the correlation
correlation_matrix= numerical_columns.corr()
print("\n---------correlation ----------\n")
print(correlation_matrix)

print("\n----------heatmap for show the correlation----------\n")
#Set a larger figsize
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.3)
plt.title('Correlation Heatmap')
plt.show()

plt.scatter(dataset['2023 population'],dataset['2000 population'])
plt.title("Correlation between 2000 population and 2023 population column")
plt.xlabel("2023 Population")
plt.ylabel("2000 Population")
plt.show()


# World population between years of (2000 - 2023)
# find the total population for each year
total_population = []
for x in years:
    total_population.append(dataset[x].sum())

print("\n----------populations between year of (2000 - 2023)----------\n ")
for i in range(len(years)):
    print(years[i], " : ", total_population[i])

plt.figure(figsize=(10, 6))
plt.plot(years, total_population, marker='o', linestyle='-', color='blue')
plt.xlabel('Years')
plt.ylabel('World Population')
plt.title('World Population Trend (2000 - 2023)')
plt.show()


print("\n-------Population Increase by Year (2000 to 2023) ")
for i in range(1, len(total_population)):
     increase = total_population[i] - total_population[i - 1]
     print(f"{years[i]} - {years[i - 1]} : {increase}")

# increase population between the year of 2000 to 2023
total_population_2000 = dataset['2000 population'].sum()
total_population_2023 = dataset['2023 population'].sum()
increase_population = total_population_2023-total_population_2000
print("\n ------ between year of 2000 and 2023 the population increase by ", increase_population)



# Find the total population by continent
continents = dataset['continent'].unique()
total_population_by_continent = []

for continent in continents:
   total_population_by_continent.append(dataset[dataset['continent'] == continent][years].sum().sum())

# Print the total population by continent
print("----Total population by continent----")
for i in range(len(continents)):
    print(continents[i], " : ", total_population_by_continent[i])

# Plotting a pie chart
plt.figure(figsize=(10, 6))
plt.pie(total_population_by_continent, labels=continents, autopct='%1.1f%%', colors=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'lightpink'])
plt.title('Population Distribution by Continent')
plt.legend()
plt.show()



# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))

axes = axes.flatten()


for i, year in enumerate(years):
    # Find the top five countries with hightes population for each year
    top_five_countries = dataset[['country', year]].nlargest(5, year)

    # Plot the total population for the top five countries
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'lightpink']
    axes[i].bar(top_five_countries['country'], top_five_countries[year], color=colors)
    axes[i].set_title(f'Top 5 Countries in {year}')
    axes[i].set_xlabel('Country')
    axes[i].set_ylabel('Total Population')


plt.tight_layout()
plt.show()


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))
axes = axes.flatten()

for i, year in enumerate(years):
    # Find the five countries with the lowest population for each year
    lowest_five_countries = dataset[['country', year]].nsmallest(5, year)

    # Plot the total population for the six countries
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'lightpink', 'lightgray']
    axes[i].bar(lowest_five_countries['country'], lowest_five_countries[year], color=colors)
    axes[i].set_title(f'Bottom 5 Countries in {year}')
    axes[i].set_xlabel('Country')
    axes[i].set_ylabel('Total Population')

plt.tight_layout()
plt.show()

dataset['growth rate'] = dataset['growth rate'].str.rstrip('%').astype('float')

# Sort the dataset based on the (growth rate) column
sorted_data = dataset.sort_values(by='growth rate', ascending=False)

# Select the top 10 rows
top_10_growth_countries = sorted_data.head(10)
top_10_less_growth_countries = sorted_data.tail(10)
# Display the result
print("\nTop 10 Countries with Highest Growth Rates \n",top_10_growth_countries[['country', 'growth rate']])
print("\nTop 10 Countries with lowest Growth Rates \n",top_10_less_growth_countries[['country', 'growth rate']])

plt.figure(figsize=(10, 8))
plt.barh(top_10_growth_countries['country'], top_10_growth_countries['growth rate'], color='lightgreen')
plt.xlabel('Growth Rate (%)')
plt.title('Top 10 Countries with Highest Growth Rates')
plt.grid(axis='x', linestyle='--')
plt.show()

plt.figure(figsize=(10, 8))
plt.barh(top_10_less_growth_countries['country'], top_10_less_growth_countries['growth rate'], color='lightgreen')
plt.xlabel('Growth Rate (%)')
plt.title('Top 10 Countries with lowest Growth Rates')
plt.grid(axis='x', linestyle='--')
plt.show()


#Predection about future population

growth_rate_2010_2015 = 441024761
growth_rate_2015_2020 = 414440842

growth = (growth_rate_2010_2015 + growth_rate_2015_2020)/2

popuplation_2023 = dataset['2023 population'].sum()

future_population_2028 = popuplation_2023 +growth
future_population_2033 = popuplation_2023 +growth +growth

print("\n 2023 Population : ", popuplation_2023)
print("\n population 2028 : ",future_population_2028)
print("\n population 2033 : ",future_population_2033)
