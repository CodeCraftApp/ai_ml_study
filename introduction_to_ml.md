# Introduction to Machine Learning

## Objectives
- Understand the different types of machine learning
- Understand the key concept of supervised machine learning
- Learn **how** solving problems with machine learning is different from traditional approaches

## What is Machine Learning?

It is a process of training a piece of software, called a model, to make useful predictions or generate content from data.

> **Example**: Suppose we want to predict the weather. There are two approaches, one traditional and another using machine learning.
> 
> In the traditional approach we would create a physics-based representation of Earth's atmosphere and surface, computing massive amounts of fluid dynamics equations. It is difficult.
> 
> Using ML, we would give an ML model an enormous amount of weather data until the ML model eventually ***learns*** the mathematical relationship between weather patterns that produce rain. We would give the model current weather data, and it would predict the amount of rain.

## Applications of Machine Learning

- Powers most important technologies like translation apps to autonomous vehicles
- Provides a new way to solve problems, answer complex questions, and create new content
- Can predict weather, estimate travel time, recommend songs, auto-complete sentences, summarize articles, and even generate never-before-seen images

## Types of Machine Learning Systems

This list of types is differentiated based on *how they learn to make predictions* or generate content:

- Supervised learning
- Unsupervised learning
- Reinforcement learning
- Generative artificial intelligence

## Supervised Learning

Supervised learning models can make predictions after seeing a lot of data with correct answers and then discovering the connections between the elements in the data that produce the correct results. This is just like a student learning a new skill/material by studying old exams that contain both the questions and the answers. Once the student is trained on enough old exams, the student is well prepared to take a new exam.

> **Note**: These ML systems are *supervised* meaning that humans give the ML system data with known correct results.

Two common applications of supervised learning are regression and classification.

### Regression

A regression model predicts a numerical value.

**Example**: A weather model that predicts the amount of rain in inches, centimeters, or millimeters is a regression model.

| Scenario | Possible Input Data | Numeric Prediction |
|----------|-------------------|-------------------|
| Future house price | Square foot, zip code, number of bedrooms and bathrooms, lot size, interest rate, property tax rate, construction cost, and number of homes for sale in the area | Price of the home |
| Future ride time | Historical traffic conditions, distance from destination and weather conditions, locality, types of road, number of speed breakers | Time in minutes and seconds to arrive at a destination |

### Classification

Classification models predict the likelihood that something belongs to a category. Unlike the regression model which outputs a numerical value, classification models output a value that states whether or not something belongs to a particular category or class.

**Example**: Classification models are used to predict if an email is `spam` or `no spam`.

Classification models are divided into two groups:
- **Binary classification models** - outputs a value from a class that contains only two values (example: `rain` or `no rain`)
- **Multi-class classification models** - outputs a value that contains more than two values (example: `rain`, `hail`, `snow`, or `sleet`)

## Unsupervised Learning

An unsupervised learning model aims to identify groups or meaningful patterns in a dataset.

For example, many unsupervised learning models use *clustering* (grouping data into *clusters*) to organize similar data.

The clustering is different from classification because the categories are not provided by us; the machine learning model itself categorizes data into relevant clusters. For example, on weather data, an unsupervised learning model may cluster data based on temperatures and then further classify seasons based on the temperature clusters.

## Reinforcement Learning

Reinforcement learning models make predictions based on getting rewards or penalties based on the actions taken by them in the environment. A reinforcement learning system generates a policy that defines the best strategy to maximize the rewards.

For example, reinforcement learning models are used to train robots to perform tasks like walking in a room and training AlphaGo to play the game of Go.

## Generative Artificial Intelligence

Generative AI is a class of models where it creates content from user input. For example, generative AI can create unique images, music, jokes; summarize articles, write new articles, or edit a photo or video.

Generative AI can take a variety of inputs and create a variety of outputs. For example, it can take text and create an image from it, or take text and images and create a video from it.

We can describe the type of generative AI model as `type-of-input`-to-`type-of-output`. For example:
- Text-to-text
- Text-to-image
- Text-to-speech
- Text-to-video
- Speech-to-text
- Image & text-to-video

| Model | Input | Example Output |
|-------|-------|-----------------|
| Text-to-text | Who invented the Le Mans race? | The 24 Hours of Le Mans was founded by the Automobile Club de l'Ouest (ACO), a motoring organization founded in 1906. The first race was held on 26 and 27 May 1923 on the Sarthe Circuit, a 13.426-kilometre (8.345-mile) road circuit located near the town of Le Mans in France. The race was originally conceived as a way to promote the ACO and the Sarthe region. Source: Bard |
| Text-to-image | An alien octopus floats through a portal while reading a newspaper. | [Image generated by AI] |
| Text-to-video | A photorealistic teddy bear is swimming in the ocean at San Francisco. The teddy bear goes under water. The teddy bear keeps swimming under the water with colorful fishes. A panda bear is swimming under water. | [Video generated by AI] |
| Text-to-code | Write a Python loop that loops over a list of numbers and prints the prime numbers. | `for number in numbers:` `  # Check if the number is prime.` `  is_prime = True` `  for i in range(2, number):` `    if number % i == 0:` `        is_prime = False` `        break` `  # If the number is prime, print it.` `  if is_prime:` `    print(number)` |
| Image-to-text | [Image of a flamingo] | This is a flamingo. They are found in the Caribbean. Source: Google DeepMind |

### How Does Generative AI Work?

Generative models learn patterns in the data with the goal of producing new but similar data. For example, you can think of generative models as:
- A comedian who learns to mimic others and pick up nuances and style of others, or writes jokes a particular way
- An artist who learns to paint in a particular style by seeing and studying lots of paintings in that style

To produce new yet similar outputs, generative models are trained using unsupervised learning and in later stages they are trained with supervised and reinforcement learning on specific sets of data related to the tasks that the model might be asked to perform (for example, summarize an article or edit a photo).

> **Note**: Generative AI is quickly advancing as new use cases are coming up all the time. For example, businesses are using models to produce good quality images for their products by removing the background and improving the quality of low-resolution images.

## Supervised Learning in Depth

Supervised learning's tasks are well defined.

### Foundational Supervised Learning Core Concepts

- Data
- Model
- Training
- Evaluating
- Inference

### Data

Data is the driving force behind ML. Data can be of any form like words, text, numbers stored in a table, values of pixels in images, or waveforms in audio.

Datasets are made up of features and labels. Features are the values that the model uses to predict the label. For example:

In a weather model that predicts rainfall, the features could be *latitude*, *longitude*, *temperature*, *humidity*, *cloud coverage*, *wind direction*, and *atmospheric pressure*. The label would be *rainfall amount*.

Unlabeled examples contain features but no label. After you create a model, the model predicts the label from the features.

#### Dataset Characteristics

A dataset is characterized by its `size` and `diversity`.

- **Size** indicates the number of examples
- **Diversity** indicates the range of examples covered

> **Note**: Good datasets are both large and diverse.

Datasets can be large and diverse, or large but not diverse, or diverse but small. In other words, a large dataset does not guarantee diversity and a diverse dataset does not guarantee many examples.

For instance, a dataset might contain 100 years worth of data, but only for the month of July. Using this dataset to predict rainfall in January would produce poor predictions.

Conversely, a dataset might cover only a few years but contain every month. This dataset might produce poor predictions because it doesn't contain enough years to account for variability.

A dataset can also be categorized based on the number of features. For example, a weather dataset may contain hundreds of features like satellite imagery and cloud coverage, while another dataset may only have 3 or 4 features like humidity, precipitation, and temperature. Datasets with more features help a model discover more patterns and help predict better. However, this is not always true because there would be features which do not causally relate to the label.

### Model

In supervised learning, a model is a complex collection of numbers that define a mathematical relationship between the specific input pattern and the output label value.

The model discovers these relationships through training.

### Training

Before a supervised learning model can predict, it must be trained. To train a model, we give a dataset to the model with actual values as well. The model's goal is to work out or get the best solution for predicting the labels from the features.

The model finds the best solution by comparing the predicted value with the actual value. Based on the difference between the predicted value and the actual value—defined as `loss`—the model gradually updates its solution. In simpler words, the model learns the mathematical relationship between the features and the label so that it can make the best prediction on the data.

For example, the model predicted `115 mm` of rain but the actual rain was `75 mm`. The model then modifies its solution so that the output is close to `75 mm`. After the model has looked upon all the examples from the dataset, it arrives at a solution that makes the best prediction, on average, for each example.

#### Steps for Training the Model

1. **Model takes in a single labeled example and provides a prediction**
   - [Training step 1 visualization]

2. **Model compares its predicted value with the actual value and updates the solution**
   - [Training step 2 visualization]

3. **The model repeats this process for each labeled example in the dataset**
   - [Training step 3 visualization]

In this way, the model learns the correct relationship between the features and the label. The gradual understanding is why large and diverse datasets produce better predictions (i.e., produce better models). The model has seen more data with a wider range of values, hence it has refined its understanding of the relationship between the features and the label.

> **Note**: During training, ML practitioners can make subtle adjustments to the configurations and features the model uses for prediction.
> 
> For example, certain features have more predictive power than others. Therefore, ML practitioners decide which features the model uses during training.
> 
> For instance, suppose a weather dataset contains `time_of_day` as a feature. An ML practitioner can add or remove `time_of_day` during training to see whether the model makes better predictions with or without it.

### Evaluating

We evaluate a trained model based on how well it learned.

When we evaluate a model, we use a labeled dataset, but we only give the features to the model. We then compare the model's prediction to the actual label values.

> **Note**: Depending on the model's evaluation, we might do more training before deploying the model in a real-world application.

### Inference

Once we're satisfied with the results from evaluating the model, we can use the model to make predictions, called **inferences**, on unlabeled examples.

In the weather app example, we would give the model the current weather conditions like temperature, atmospheric pressure, and relative humidity, and it would predict the amount of rainfall.

---

## References

### Key Terms on This Page

- Classification model
- Clustering
- Model
- Policy
- Prediction
- Regression model
- Reinforcement learning
- Reward
- Supervised learning
- Training
- Unsupervised learning
- Example
- Feature
- Inference
- Labeled example
- Label
- Loss
- Prediction
- Training

### Next Steps

- **Machine Learning Crash Course** - If you're ready for an in-depth, hands-on approach to learning more about ML.
- **Problem Framing** - If you're looking for a field-tested approach for creating ML models and avoiding common pitfalls.
- **People + AI Guidebook** - If you're looking for practical guidance for designing human-centered AI products.
