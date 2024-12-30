#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>

//this macros are for our data set iris.csv for now
#define MAX_ROWS 150
#define MAX_FEATURES 4  

typedef struct svm_node{
    int index; // feature index, ends -1
    double value; // feature value
}svm_node;

typedef struct svm_problem{
    int len; // numbers of features
    double *y; // list of labels {-1,1}
    svm_node **x ; // each datapoint features(array of features)
}svm_problem;

typedef struct svm_parameters{
    double C;       // regularization parameter
    double eps;     // stopping tolerance 
    double eta;     //learning rate
    int max_iter;   //number of training iterations
}svm_parameters;

typedef struct svm_model{
    double *w;      // weight vectore of length MAX_FEATURES
    double b        // bias term

}svm_model;


/*
The decision function:
f(x) = w·x + b
*/
double decision_function(svm_model *model, svm_node *x){
    return dot_product(model->w,x) + model->b;
}


/*
    svm_predict fonction:
    will return : +1 if the decision fonction is >= 0 else -1
*/

double svm_predict(svm_model *model, svm_node *x){
    double val = decision_function(model,x); // x is the one data point
    
    if(val >= 0.0) return 1.0;
    else           return -1.0;
}