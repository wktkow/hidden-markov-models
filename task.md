# Topics in Advanced Modeling Techniques

**Assignment:** Hidden Markov Models

**Assignment date:** December 12, 2024

**Due date:** January 10, 2025

---

## 1 Introduction to the power-consumption dataset

The dataset contains 1,416 hourly means of power consumption of a city in the period of 01/01/2017 to 28/02/2017.

## 2 Data file

*   Comma-separated file z1_hr_mean.csv
*   Variables:
    1.  `day_in_year`: day (format: DD/MM/YYYY)
    2.  `hr`: hour of the day
    3.  `mean_power`: mean power consumption (in KWh)
    4.  `sqe`: sequence number

## 3 Assignment

*   The response of interest is the mean power consumption.
*   Fit HMMs with different numbers of states to the data. Interpret the models. Check their fit.
*   Which model would you select/recommend? Why?
*   In your report, provide the information (printouts, figures, etc.) supporting your decisions and conclusions. Include (in an appendix of max. 1 page) the main parts of the syntax underlying your results. Limit your report to 8 pages, including the title page and the appendix. Submit your report on BlackBoard by the indicated deadline.

## 4 Software hints

*   You can use any software of your choice, but you have got to know and understand the methods that are implemented in the software. A possible tool is the `HiddenMarkov` R-package. An HMM is defined by using function `dthmm`. The model can be fitted by using the `BaumWelch` or `neglogLik` functions. Global decoding can be performed by using function `Viterbi`. The help of the latter function shows how to perform local decoding.
