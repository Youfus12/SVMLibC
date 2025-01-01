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

double dot_product(const double *w,const svm_node *x){

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
double decision_function(const svm_model *model, const svm_node *x){
    return dot_product(model->w,x) + model->b;
}


/*
    svm_predict fonction:
    will return : +1 if the decision fonction is >= 0 else -1
*/

double svm_predict(const svm_model *model,const svm_node *x){
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
    load dataset:
    Labels : 0 / 1 , will be mapped +1 -1
    Each Row contains : Four features and a class label
    Will ignore the  Header
*/
int load_dataset(const char *filename, svm_problem *prob, int max_rows)
{
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        return -1;
    }

    // Initialization of the svm problem:
    prob->len = 0; // Initialize the number of data points
    prob->y = (double *)calloc(max_rows, sizeof(double));           // array of labels size len
    prob->x = (svm_node **)calloc(max_rows, sizeof(svm_node *));    // array of pointers to a feature vector

    if (!prob->y || !prob->x) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        return -1;
    }

    char line[256]; // buffer will read 256 characters each row max, and it's enough.

    // Attempt to read header line
    /*
        fgets(buffer, size to read, file to read) if it's able to read will return true, else false
    */
    if (fgets(line, sizeof(line), file)) {
        // The buffer read the header line and ignored it, now will go to the next line
    }

    while (fgets(line, sizeof(line), file) && prob->len < max_rows) { // or feof(file), if true this means we are at the end of the file
        double f1, f2, f3, f4;
        int target;

        /*
            sscanf: Reads formatted input from a string (line) and assigns the extracted values to the provided variables.
            Expecting 5 fields:
            "%lf,%lf,%lf,%lf,%d" : the data should be in this format 
                lf: double
                ,: comma
                d: int +1, -1
        */

        int fields = sscanf(line, "%lf,%lf,%lf,%lf,%d", &f1, &f2, &f3, &f4, &target);
        if (fields != 5) {
            fprintf(stderr, "Skipping invalid line: %s\n", line);
            continue; // If the line doesn't match the format, skip it
        }

        if (target != 0 && target != 1) {
            // Skip lines that aren't binary 0/1
            fprintf(stderr, "Skipping line with non-binary label: %s\n", line);
            continue;
        }

        double label = (target == 0) ? 1.0 : -1.0; // if label of the sample is 0, return 1.0, else -1.0

        // Allocate feature array for 4 features plus a terminator
        svm_node *x_node = (svm_node *)malloc((MAX_FEATURES + 1) * sizeof(svm_node)); // Allocate for 4 features + 1 terminator
        if (!x_node) {
            fprintf(stderr, "Memory allocation failed for features\n");
            break; // Exit if memory allocation fails
        }

        x_node[0].index = 1; x_node[0].value = f1;
        x_node[1].index = 2; x_node[1].value = f2;
        x_node[2].index = 3; x_node[2].value = f3;
        x_node[3].index = 4; x_node[3].value = f4;
        x_node[4].index = -1;  // Terminator, that's why we added 1 in max features

        prob->y[prob->len] = label; 
        prob->x[prob->len] = x_node;
        prob->len++; // Increment the number of data points
    }

    fclose(file); // Close the file when done
    return prob->len; // Return the number of data points loaded
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
svm_model *svm_train(const svm_problem *prob,const svm_parameters *param){
    //TIP: the prob here is the training dataset

    // INITIALIZATION OF THE MODEL:
    svm_model *model = (svm_model*)malloc(sizeof(svm_model));
    model->w = (double*)calloc(MAX_FEATURES,sizeof(double)); //w is a vector of max_features size of doubles, and initialize them with 0 using callco
    model->b = 0.0;

    double C = param->C;        // Regularization parameter
    double eta = param->eta;    // Learning Rate
    int max_iter = param->max_iter;
    int l = prob->len;


    // Initialize weights to zero
    for(int d = 0; d < MAX_FEATURES; d++) {
        model->w[d] = 0.0;
    }
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

double calculate_accuracy(const svm_model *model, const svm_problem *test_prob) {
    int correct = 0;
    for(int i = 0; i < test_prob->len; i++) {
        double pred = svm_predict(model, test_prob->x[i]); // predict the label with the test samples
        if((int)pred == (int)test_prob->y[i]) {             // compare the predicted label with the test label
            correct++;
        }
    }
    return ((double)correct / test_prob->len) * 100.0;
}

void svm_free_model(svm_model *model)
{
    free(model->w);
    free(model);
}
