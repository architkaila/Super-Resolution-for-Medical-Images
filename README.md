# Super Resolution for Medical Images
> #### _Archit | Spring '23 | Duke AIPI 540 Final Project_
&nbsp;

## Project Description
From our perspective, there are two types of people when it comes to cooking: those who see it as a fun activity and a hobby of theirs and those who see it as a chore they need to do for themselves or their family each day. If you are in the former group, you likely enjoy trying out new recipes often. Unfortunately, choosing which recipe to try next is often the hardest part. There are so many unique and delicious recipes out there, and it can be very difficult to narrow them all down to one or two that you want to try next. If you view cooking as a chore, you likely have a small set of meals that you cook over and over and over again. These can become stale and boring over time, and adding in a few new meals once in a while is a great way to spice things up. Either way, if you find yourself in need of some new recipes, we have just the solution for you!

We have created a recipe recommendation system that allows you to get a series of recipes that align with your preferences in no time. Using information such as how long the meal takes to make, the nutritional value, and the ingredients present in the dish, we utilize content filtering to recommend recipes similar to those of meals that you already know you like. As a new user, you simply select what meals and dishes you enjoy / think look good and then we recommend the best recipes for you to try next.


&nbsp;
&nbsp;
## Running the demo (StreamLit)
The Streamlit-based web application is hosted live and can be accessed [here](https://food-whisperer.streamlit.app/?page=Recipe+Recommender)

&nbsp;
&nbsp;
### Follow the steps to setup and run the app locally:
**1. Clone this repository and switch to the streamlit-demo branch**
```
git clone https://github.com/guptashrey/Food-Whisperer
git checkout st
```
**2. Create a conda environment:** 
```
conda create --name environ python=3.9.16
conda activate environ
```
**3. Install requirements:** 
```
pip install -r requirements.txt
```
**4. Run the application**
```
streamlit run streamlit_app.py
```
**5. StreamLit Appication:**

&nbsp;
#### Select a few recipe's you love and click the 'Recommend Recipies' button to see the magic
>![img.png](data/images/dashboard1.png)

&nbsp;
#### The app would recommend you some recipies using content filtering recommendation system
>![img.png](data/images/dashboard2.png)

&nbsp;


