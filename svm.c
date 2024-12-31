#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//this macros are for our data set iris.csv for now
#define MAX_ROWS 150
#define MAX_FEATURES 4  

typedef struct svm_node{
    int index; // feature index, ends -1
    double value; // feature value
}svm_node;

typedef struct svm_problem{
    int len; // numbers of data points
    double *y; // list of labels {-1,1}
    svm_node **x ; // array of pointers : that each pointer point to an array of features => this is an array of arrays features
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