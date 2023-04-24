# flat_price
**Overview**

**Data**

The original dataset is from kaggle.com, it contains information about houses’ price, location, square meters, number of rooms, building date and so on. Generally, there is all the necessary information to build the recommendation system for estimating the price of houses. 

**Processing data**

The task was to build a recommendation model for estimating houses' price in the future which provides the least RMSLE error. The behavior of RMSLE error is close to MSLE error, that is why in this project MSLE error was estimated.

For simplifying the metric analysis not just the houses’ price was used, but its logarithm.
In original data, there are three types of features: qualitative, quantitative and absent data (None). That is why at the beginning the data processing was made.

*Quantitative data preprocessing*

Firstly, quantitative None values were replaced with the mean value of the feature. Then for decreasing the number of features columns with high correlation were deleted. The threshold value for correlation was 0.9. The third step is to delete constant or quasi-constant values, for that aim the variance was estimated and the threshold value was chosen as 0.1.

*Qualitative data processing*

There are a lot of methods for transforming qualitative data into quantitative, but I used just One-Hot-Encoding (OHE) and Mean-Target-Encoding (MTE) because they illustrated quite a good error at the end.

All qualitative data, except "datatime" property, were modified to quantitative data using OHE method if there are fewer than five unique features, and MTE method if there are more of them. The “datatime” property was divided by day, month, and year, and after that month and year features were modified using OHE method, days were deleted because days’ alteration doesn’t affect price significantly.

*Extra processing*

Moreover, in this project peak values were deleted and the classification based on primary and secondary housing was made. These techniques slightly improved the error, but sometimes it is important to save peak values, for example, in the case of estimating homes for extra luxury or unordinary customers. That is why preprocessing data doesn’t contain filtration of peak values, it was made during the building of the model.

**Model**

In this case, the aim is to predict the future cost of houses, so the standard method of cross-validation doesn’t work. The training set for this model is based on the time serial split algorithm the smallest amount of splits is four because there are four years in the dataset. 

![Illustration of timeSeriasSplit method for cross-validation](https://user-images.githubusercontent.com/104028421/232858510-07355817-12d4-4707-8a6a-016e7485daf8.png)


Moreover, to overcome overfitting and improve model the regularization and scaling were added. In this project, Losso regularization was used. Data scaling was implemented only on training data

As it was mentioned before, the segmentation of data based on primary and secondary housing was used. That is why there are two extra models for each case.
