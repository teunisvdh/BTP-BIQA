# BTP-BIQA

*The files in this repository were used during the master thesis 'Explainable AI for Blind Image Quality Assessment'.* 

![Alt Text](https://github.com/teunisvdh/BTP-BIQA/blob/master/doc/Application%20Thesis%20Images%20Flower.png)

##### Research findings

1.	Deep learning-based BIQA models are powerful, for the reason that they facilitate end-to-end-learning as well as content aware quality assessment. 

2.	In this exploratory project two decision trees were developed to assess technical image quality (subjective) as well as aesthetic image quality (highly subjective). Although the technical quality decision tree performs comparable to conventional BIQA methods, performance of both decision trees is lower than that of state-of-the-art deep learning-based methods. 

3.	By developing a web application, I have shown that the out-of-the-box characteristics of a decision tree are useful in making the algorithm transparent, explainable and adaptable.

4.	Survey results of users who interacted with the web application, hint at three conclusions:
    *	Perceived model performance cannot be fully explained by its test set performance; 
    *	Perceived performance is not the only characteristic influencing model preference;
    *	Users prefer a model that is transparent, explainable and adaptable to their needs.

##### Main contents of this repository 
| File | Description |
| --- | --- |
| Clean Quality Decision Tree.ipynb | Training the technical quality decision tree by extracting 16 features. | 
| Clean Aesthetics Decision Tree.ipynb | Training the aesthetic quality decision tree by extracting 23 features. |
| Aesthetics Dataset Web Scraper.ipynb | Compiling a dataset based on dpchallenge.com, the original source of AVA, in order to control image content based on several keywords (social events). |
| deploy (folder) | Final web application. Additional languages: JavaScript, CSS and HTML. Feature extraction functions can be found in helpers.py. This folder originally included KonCept224 and the kutils package introduced by Hosu et al. (2020). Please download these files from the original source before running the code. |
| BTP-BIQA Experiment.ipynb | Google Colab Notebook: setting up a local server for the deploy package and sharing it using Ngrok. |
| BTP-BIQA used in app.sav | The exact configuration of the technical quality decision tree as used in the web application. |
| Survey Results Translated.csv | A machine translation of the survey results. Names/email addresses were deleted. |
