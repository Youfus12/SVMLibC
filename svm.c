#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*

    Vector of features of one data point or vector of svm_node:
                [ {1 : feature } , {2 : feature } .., {-1, null} ]
    **xi : array of pointers, each pointer point to an vector of features

*/
//this macros are for our data set iris.csv for now
#define MAX_ROWS 150
#define MAX_FEATURES 4  

typedef struct svm_node{
    int index; // feature index, (1 based, means 1,2,3,4 ..n)
    double value; // feature value
}svm_node;

typedef struct svm_problem{
    int len; // numbers of data points
    double *y; // list of labels {-1,1}
    svm_node **x ; // array of pointers : each points to a vector of svm_nodes(features)
}svm_problem;

typedef struct svm_parameters{
    double C;       // regularization parameter
    double eps;     // stopping tolerance 
    double eta;     //learning rate
    int max_iter;   //number of training iterations
}svm_parameters;

typedef struct svm_model{
    double *w;      // weight vectore of length MAX_FEATURES
    double b;        // bias term

}svm_model;

double dot_product(double *w,svm_node *x){

    double sum = 0.0;
    while(x->index != -1){
        int idx = x->index - 1 ;
        sum += w[idx] * x->value;
        x++;
    }
    return sum;
}
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

/*
    fullprob: the full dataset containing l samples,
    traindata: output parameter , to store the training portion
    test_data : output parameter to store the test portion
    test_size: value from (0.0 - 1.0) indicatin what portion will get the train data
*/
void split_dataset(svm_problem *full_prob,svm_problem *train_prob, svm_problem *test_prob, double test_precentage){

    int total = full_prob->len;
    int test_size = (int)(total * test_precentage);
    int train_size = total - test_size;

    // Shuffling part with the Knuth method shuffle: cause sometimes the dataset can be ordered by features or lables and the splitting will be illogical


    int  *indices = (int *)malloc(total * sizeof(int)); //array of ints size l

    for(int i = 0; i< total; i++){
        indices[i] = i; //sorted array from [0,1, ..., total -1]
    }

    //shuffeling using indicies array: 

    for(int i = total - 1; i>0; i--){
        int j = rand() % (i+1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }

/*
     Now we have an array of indices in a random order
     time to split the dataset

*/

    // Allocate Training:

    train_prob->len = train_size;
    train_prob->y = (double *)malloc(train_size * sizeof(double)); // array of labels
    train_prob->x = (svm_node **)malloc(train_size * sizeof(svm_node*)); //array of pointers , that point to an array of features of one datapoint(svm_node)
    for(int i= 0; i < train_size; i++){
        int idx = indices[i];
        train_prob ->y[i] = full_prob->y[idx];
        train_prob ->x[i] = full_prob->x[idx]; // affectation of a pointer that points to an vector feature: not deep copy of full_prob->x, just the pointers

    }

    // Allocate testing

    test_prob ->len = test_size;
    test_prob ->y = (double *)malloc(test_size * sizeof(double));
    test_prob ->x = (svm_node**)malloc(test_size *sizeof(svm_node*));

    for(int i = 0; i < test_size; i++){
        int idx = indices[train_size + i];
        test_prob->y[i] = full_prob->y[idx];
        test_prob->x[i] = full_prob->x[idx];
    }

    free(indices);
    // if you dont need it, free it.
}

/*
  svm_train:
   A simple stochastic Subgradient desqcent approach for linear SVM with hinge loss:
 
   Objective:  0.5 * ||w||^2 + C * Σ max(0, 1 - y_i * (w·x_i + b))
 
   Per-sample subgradient update rule:
     1) Always shrink w by factor (1 - eta)  (weight decay for regularization)
     2) If y_i * (w·x_i + b) < 1:  (margin is violated)
           w <- w + eta * C * y_i * x_i
           b <- b + eta * C * y_i
           else
           b remains the same
 
   We do this for max_iter epochs, shuffling the data each epoch.
 */
svm_model *svm_train(svm_problem *prob, svm_parameters *param){
    //TIP: the prob here is the training dataset

    // INITIALIZATION OF THE MODEL:
    svm_model *model = (svm_model*)malloc(sizeof(svm_model));
    model->w = (double*)calloc(MAX_FEATURES,sizeof(double)); //w is a vector of max_features size of doubles, and initialize them with 0 using callco
    model->b = 0.0;

    double C = param->C;        // Regularization parameter
    double eta = param->eta;    // Learning Rate
    int max_iter = param->max_iter;
    int l = prob->len;

    //Shuffeling: Cause the Stochastic gradientDecent preforms better if we randomize the order of the samples

    int *indices = (int*)malloc(l * sizeof(int));
    for(int i = 0; i < l; i++) indices[i] = i;

    // PRocess the entire data set point to point : iteration = epoch

    for(int epoch = 0; epoch < max_iter-1; epoch++ ){

        // A: shuffle training points each epoch:
        for(int i = l-1; i>0 ; i--){
            int j = rand() % (i+1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

        // B: Iterate over each datapoint in the training with shuffeled order
        for(int i = 0; i < l; i++){
            int idx = indices[i];       //Sample od dataPoint
            double yi = prob->y[idx];   // its label +1 or -1
            svm_node *xi = prob->x[idx];// its pointer that point to the vector features , and points to the first value of the vector features

            // Training Phase for the weights and bias

            /*
                Step1: Regularization :
                shrink all w values or Penalazing large weigths by shrinking them to 0:
             */

            for(int d = 0; d < MAX_FEATURES; d++){
                model->w[d] = model->w[d] * (1-eta); 
            }

            /*
                Step 2:  check the margin
                margin :  y_i * (w·x_i + b) 
            */
            double margin = yi* (dot_product(model->w, xi) + model->b);
            /*
            margin <1 violated => hingeloss subgradient
                w <- w + eta * C * y_i * x_i
                b <- b + eta * C * y_i
            */
           if(margin < 1){
            while(xi->index != -1){
            int d = xi->index - 1; // To  0 based indexing
            model->w[d] += eta * C * yi * xi->value; // Dont forget xi is a pointer of vector features and its pointing to the first value of it
            xi++;
              }
            model -> b += eta * C * yi;        
           }
            // else (margin >= 1) => no hinge update

        }

    }
    
    
    free(indices); // Good job indices[], now be Free.

    return model;

}

void svm_free_model(svm_model *model)
{
    free(model->w);
    free(model);
}
