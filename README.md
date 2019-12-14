# Avatar or Mortal Engines? Predicting Movie Profitability from Scripts
As mentioned in the writeup, our project had two stages: Trial 1, in which we focused on predicting profitability of a movie solely based on its full script; and Trial 2, in which we predicted "Freshness" of a movie based on its synopsis and other non-textual features.   
Below you can find the commands that replicate our results.

All commands were run in Python 3.8. All packages required can be found in the requirements.txt.

The following Python packages are required:

appnope==0.1.0  
backcall==0.1.0  
cycler==0.10.0  
Cython==0.29.14  
decorator==4.4.1  
fuzzywuzzy==0.17.0  
imbalanced-learn==0.5.0  
imblearn==0.0  
ipykernel==5.1.3  
ipython==7.9.0   
ipython-genutils==0.2.0   
jedi==0.15.1  
joblib==0.14.0  
jupyter-client==5.3.4  
jupyter-core==4.6.1  
kiwisolver==1.1.0  
matplotlib==3.1.2  
mlxtend==0.17.0  
nltk==3.4.5  
numpy==1.17.4  
pandas==0.25.3  
parso==0.5.1  
pexpect==4.7.0  
pickleshare==0.7.5  
prompt-toolkit==2.0.10  
ptyprocess==0.6.0   
Pygments==2.4.2  
pyparsing==2.4.5  
python-dateutil==2.8.1  
python-Levenshtein==0.12.0  
pytz==2019.3  
pyzmq==18.1.0   
scikit-learn==0.22.dev0  
scipy==1.3.2  
seaborn==0.9.0  
six==1.13.0  
tornado==6.0.3  
traitlets==4.3.3  
wcwidth==0.1.7  

## Trial 1
Feature extraction - python trial_1_feature_extraction.py  
Training - python trial_1_training.py  
 
## Trial 2
Feature extraction - python trial_2_feature_extraction.py  
Training - python trial_2_training.py  

## Other files
file_titles.txt - used for file matching   
script_rating_correlations.ipynb, correlation_analysis.ipynb - notebooks for script analysis   
BERT.ipynb - notebook for BERT analysis via ktrain  
helper_scripts/ - scripts for matching   
datasets/ - datasets used  
movie_scripts/ - movie scripts    
pickled_files/ - pickled files for caching consistent data
statified_analysis.ipynb - attempts at stratified analysis of models
correlation_analysis.ipynb - used to generate data plots

Note: If there are path issues, it is likely because a file has been moved to the datasets folder. Remove datasets that are not found and place them in the home directory in order to fix this.
