# Graduation-Design
Here is the Graduation Design for 2024 Leeds graduation!

## The introduction of this project
This project is reproduced by [SpectralNet Resource](https://github.com/kstant0725/SpectralNet)
And just use this model to do some experiment.
Tips: This is Pytorch version


## Here are structure of this portfoilo
```
Other Normal Algorithm
SpectralNet
environment.yml
```

The `environment.yml` file is configuration of the virtual environment. It is suitable for all code in this project.

In `Other Normal Algorithm`, there are some .py file which could execute the Basic K-means algorithm and spectral clustering algorithm by sk-learn library. If you want to use these codes to do something, you should enter these file and programme the code which could load your dataset.

In `SpectralNet` is the model. You could run this project by command `python main.py config/` + any config file. You could write your own config file, and its format could be modelled by reuters.json.

Tips: for any config files, you should update the function of loading the dataset in `data.py`.

## Other tips
The evaluation functions could loaded in `metrics.py`. And use it straightly in model. Except that, the `SoectralNetBlog.ipynb` is the Jupyter version for this model. But you also need the same way to change the original code so that it could execute youself dataset.
