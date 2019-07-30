# Machine Learning Classification Model for Korean Names
We are building a classification model to help us determine whether a name is Korean or non-Korean. 

## Getting Started

Previously, we relied on considerable manual labor and intuitive speculations in order to distinguish whether an incident involved or was applicable to Korean population. The project's primary purpose is to more accurately and efficiently accomplish the same task.


### Acquiring Sample Data

We prepared our sample data by combining common first and last names for both Korean and Non-Korean names. These are the sources for the common first and last names used: 

* [Korean Last Name](https://www.familyeducation.com/baby-names/browse-origin/surname/korean) - FamilyEducation ("Korean Last Names")
* [Korean First Name](https://en.wikipedia.org/wiki/List_of_the_most_popular_given_names_in_South_Korea) - Wikipedia ("List of the most popular given names in South Korea")
* [English Last Name](https://rometools.github.io/rome/) 
* [English First Name](https://rometools.github.io/rome/) 


### GitHub Repo Reference

* **Repository Name**: deep-name-classifer
* **Repository Description**: "A deep learning keras based python script that predicts the nationality origin of a person from the name sequence. It classifies a name as native or foreign. It is based on Indian context of naming a person."
* **Libraries Used**: numpy, pandas, keras
* **Repository Link**: [Github](https://github.com/vivek7266/deep-name-classifier/blob/master/deep-name-classifier.py)


## Description

Components:
1. sample_with_numbers.py - Sample names from csv files
    * 200,000 rows of Non-Korean Names
    * 27,710 rows of Korean Names
2. nc.py - Python file to train model from a csv file
3. load_model.py - Python file to load trained model and test names

Model:
1. LSTM (3 layers, 32*32*2)
2. Dropout = 0.15
3. Loss function = Binary cross entropy


## Future Directions

1. We may use a web-crawler to search for Korean names or related topics. 

## Contributors
* Roy Seungro Lee
* Miyoun Song

## Screenshots:

### Sampling Names:
<img src="/img/name_sample.png" width="60%" height="50%">

## Training Model:
<img src="/img/train_model.png">

## Load Model and Results:
<img src="/img/load_model.png" width="60%" height="50%">
