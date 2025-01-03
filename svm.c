#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
/*

    Vector of features of one data point or vector of svm_node:
                [ {1 : feature } , {2 : feature } .., {-1, null} ]
    **xi : array of pointers, each pointer point to an vector of features

*/
//this macros are for our data set iris.csv for now
#define MAX_ROWS 1000
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


//All the fonction declaration , later ill transport them in a header file
double dot_product(const double *w, const svm_node *x);
double decision_function(const svm_model *model , const svm_node *x);
double svm_predict(const svm_model *model, const svm_node *x);
void split_dataset(svm_problem *full_prob, svm_problem *train_prob, svm_problem *test_prob, double test_percentage);
int load_dataset(const char *filename, svm_problem *prob, int max_rows, int selected_features[]);
svm_model *svm_train(const svm_problem *prob, const svm_parameters *param);
double calculate_accuracy(const svm_model *model, const svm_problem *test_prob);
void svm_free_model(svm_model *model);
int write_plot_data(const svm_problem *prob, const svm_model *model);
int plot_data(FILE *gnuplot_pipe, int feature1, int feature2);
void free_dataset(svm_problem *prob, int free_x_nodes);
void cleanup_temp_files();
//--------------------------------------------------------------------------------

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
void split_dataset(svm_problem *full_prob,svm_problem *train_prob, svm_problem *test_prob, double test_percentage){

    // Check if the test precentage is [0.0 to 1.0] range:
    if(test_percentage < 0.0 || test_percentage > 1.0){
        fprintf(stderr, "Error: test_percentage must be between 0.0 and 1.0\n");
        exit(EXIT_FAILURE);
    }
   

    int total = full_prob->len;
    int test_size = (int)(total * test_percentage);
    int train_size = total - test_size;

    // Shuffling part with the Knuth method shuffle: cause sometimes the dataset can be ordered by features or lables and the splitting will be illogical


    int  *indices = (int *)malloc(total * sizeof(int)); //array of ints size l
    if(!indices){
        fprintf(stderr, "Memory allocation failed for indices\n");
        exit(EXIT_FAILURE);
    }

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
    if (!test_prob->y || !test_prob->x) {
        fprintf(stderr, "Memory allocation failed for testing data\n");
        free(indices);
        exit(EXIT_FAILURE);
    }
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
int load_dataset(const char *filename, svm_problem *prob, int max_rows, int selected_features[])
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

        // Allocate feature array for 2 features plus a terminator
        svm_node *x_node = (svm_node *)malloc((2 + 1) * sizeof(svm_node)); // Allocate for 2 features + 1 terminator
        if (!x_node) {
            fprintf(stderr, "Memory allocation failed for features\n");
            break; // Exit if memory allocation fails
        }
        // W will remap the selected features to indices 1 and 2
        for(int s = 0; s < 2; s++){
            int feat = selected_features[s]; // contains the real indice of the feature
            x_node[s].index = s +1 ;
            /*
            For s = 0 (first selected feature):
            x_node[0].index = 1
            For s = 1 (second selected feature):
            x_node[1].index = 2
            */
            switch(feat){
                case 1: // if the real indice is 1 we will affect the value of the f1, first feature
                x_node[s].value = f1;
                break;
                case 2:
                x_node[s].value = f2;
                case 3:
                x_node[s].value = f3;
                case 4:
                x_node[s].value = f4;
                default:
                x_node[s].value = 0.0;
            }
        }

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

    // Checker :
    if(!model){
        fprintf(stderr, "Memory allocation failed for model\n");
        exit(EXIT_FAILURE);
    }

    if(!model->w){
        fprintf(stderr, "Memory allocation failed for weights\n");
        free(model);
        exit(EXIT_FAILURE);
    }

    // Existing training logic

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
    if(!indices){
        fprintf(stderr, "Memory allocation failed for training indices\n");
        svm_free_model(model);
        exit(EXIT_FAILURE);
    }
    for(int i = 0; i < l; i++) indices[i] = i;

    // PRocess the entire data set point to point : iteration = epoch

    for(int epoch = 0; epoch < max_iter; epoch++ ){

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
            svm_node *xi_update = prob ->x[idx]; // Reset pointer
            while(xi_update->index != -1){
                int d = xi_update->index - 1; // To  0 based indexing
                if(d >=0 && d < 2){
                     model->w[d] += eta * C * yi * xi_update->value; // Dont forget xi is a pointer of vector features and its pointing to the first value of it
                }
                xi_update++;
              }
            model -> b += eta * C * yi;        
           }
            // else (margin >= 1) => no hinge update

        }

    
    // Progress  for debugging
        if((epoch+1) % 100 == 0){
            printf("Completed epoch %d/%d\n", epoch+1, max_iter);
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

/*
    VISUALISATION PART:
    -WE will be using Gnuplot
 */

/* write_plot_data:
Generates temp files containing the cordinates of the datapoints of each class,
and the svm decision boundry (hyperplan) and the two margins

its job :
-Separates data points into two classes based on their labels.
-Calculates the SVM decision boundary (hyperplane) and its margins.
-Writes the coordinates of these elements to respective data files.

Return 0 if everything  good
return -1 if fails
*/

int write_plot_data(const svm_problem *prob, const svm_model *model){
    // File names 
    const  char *class1_file = "class1.dat"; // +1
    const char *class2_file = "class2.dat";  // -1
    const char *hyperplane_file = "hyperplane.dat"; // Stores two points defining the SVM decision boundary 
    const char *margin1_file = "margin1.dat";       // Stores two points defining the upper margin (w·x + b = 1).
    const char *margin2_file = "margin2.dat";       // Stores two points defining the upper margin (w·x + b = -1).

    // open files in write mode:

    FILE *f_class1 = fopen(class1_file,"w");
    FILE *f_class2 = fopen(class2_file, "w");
    FILE *f_hyperplane = fopen(hyperplane_file, "w");
    FILE *f_margin1 = fopen(margin1_file, "w");
    FILE *f_margin2 = fopen(margin2_file, "w");

    if(!f_class1 || !f_class2 || !f_hyperplane || !f_margin1 || !f_margin2 ){

        perror("Failed to open plot data files"); // Prints error msg
        if (f_class1) fclose(f_class1); //if its open close it
        if (f_class2) fclose(f_class2);
        if (f_hyperplane) fclose(f_hyperplane);
        if (f_margin1) fclose(f_margin1);
        if (f_margin2) fclose(f_margin2);
        return -1;

    }

    // Separation class 1 and class 2 files

    for(int i = 0; i < prob->len;i++){
        double f1 = 0.0 , f2= 0.0;
        svm_node *x = prob->x[i]; // *x pointing to an array of features

        while(x->index != -1){
            if(x->index == 1) f1 = x->value;
            else if(x->index == 2) f2= x->value;  
            x++;
        }
        // store the features of the sample in its class
        if(prob->y[i] == 1) fprintf(f_class1, "%lf %lf\n",f1,f2);
        else if(prob->y[i] == -1) fprintf(f_class2, "%lf %lf\n",f1,f2);
    }

    fclose(f_class1);
    fclose(f_class2);

    /* Here we will use the trained model to
    Calculate hyperplane: w1*f1 + w2*f2 + b = 0 => f2 = -(w1*f1 + b)/w2
    Calculate margins: w1*f1 + w2*f2 + b = 1 and w1*f1 + w2*f2 + b = -1
    */

   /*
     first we will identifie the minimum and maximum values of Feature 1 across all 
     data points to determine the range over which to plot the hyperplane and margins.

     then use it to calc the min max of f2 using the min max f1 and the trained model(w and b trained)
     for hyperplane, margin1 and margin2 lines

   */

    double f1_min = DBL_MAX, f1_max = -DBL_MAX; // dbl_max : lowest double value possible , to make sure that the feature is higher in value

    for(int i = 0; i<prob->len; i++){
        double current_f1 = 0.0;
        svm_node *x = prob->x[i];
        while(x->index!=-1){
            if(x->index == 1) current_f1 = x->value; break;
            x++;
        }

        if(current_f1 > f1_max) f1_max = current_f1;
        if(current_f1 < f1_min) f1_min = current_f1;
    }
    // Extend the range slightly for better visualization, 10% margin
    double range = (f1_max - f1_min) * 0.1;
    f1_min -= range;
    f1_max += range;

    // Ensure w2 is not zero to avoid division by zero
    if (model->w[1] == 0) {
        fprintf(stderr, "Cannot plot hyperplane or margins because w2 is zero.\n");
        fclose(f_hyperplane);
        fclose(f_margin1);
        fclose(f_margin2);
        return -1;
}   
    
    // Calcuulate f2 that is related to the f1 min and max for hyperplane For plotting it

    double f2_hyper_min = -(model->w[0]*f1_min + model->b) / model->w[1];
    double f2_hyper_max = -(model->w[0] * f1_max + model->b) / model->w[1];

    // calcultae f2 for margin :
    double f2_margin1_min = -(model->w[0] * f1_min + model->b - 1) / model->w[1];
    double f2_margin1_max = -(model->w[0] * f1_max + model->b - 1) / model->w[1];

    double f2_margin2_min = -(model->w[0] * f1_min + model->b + 1) / model->w[1];
    double f2_margin2_max = -(model->w[0] * f1_max + model->b + 1) / model->w[1]; 

    // Check for debuging
    printf("\nHyperplane Points:\n");
    printf("(%lf, %lf)\n", f1_min, f2_hyper_min);
    printf("(%lf, %lf)\n", f1_max, f2_hyper_max);

    printf("\nMargin1 Points (w·x + b = 1):\n");
    printf("(%lf, %lf)\n", f1_min, f2_margin1_min);
    printf("(%lf, %lf)\n", f1_max, f2_margin1_max);

    printf("\nMargin2 Points (w·x + b = -1):\n");
    printf("(%lf, %lf)\n", f1_min, f2_margin2_min);
    printf("(%lf, %lf)\n", f1_max, f2_margin2_max);


    // Write the the two points of hyperplane/margin1,2
    fprintf(f_hyperplane,"%lf %lf\n",f1_min,f2_hyper_min);
    fprintf(f_hyperplane,"%lf %lf\n",f1_max,f2_hyper_max);

    fprintf(f_margin1,"%lf %lf\n",f1_min,f2_margin1_min);
    fprintf(f_margin1,"%lf %lf\n",f1_max,f2_margin1_max);

    fprintf(f_margin2,"%lf %lf\n",f1_min,f2_margin2_min);
    fprintf(f_margin2,"%lf %lf\n",f1_min,f2_margin2_min);

    //if everything works
    return 0;
}

// group of comands for gnuplot and plotting the stored cordinates :
int plot_data(FILE *gnuplot_pipe, int feature1, int feature2){
    // commands to setup the plot
    fprintf(gnuplot_pipe,"set title 'SVM Decision Boundry with Margins\n");
    fprintf(gnuplot_pipe,"set xlabel 'Feature %d'\n",feature1);
    fprintf(gnuplot_pipe,"set ylabel 'Feature %d'\n",feature2);
    fprintf(gnuplot_pipe,"set grid\n");
    fprintf(gnuplot_pipe,"set key outside\n");
    fprintf(gnuplot_pipe,"set autoscale fix\n");
    fprintf(gnuplot_pipe,"set size ration -1\n"); // to fix the aspect ratio

    // Styling for cleaner visualisation
    fprintf(gnuplot_pipe,"set style line 1 lc rgb 'orange' pt 7`s 1.5 # Class +1\n");
    fprintf(gnuplot_pipe,"set style line 2 lc rgb 'black' pt 7`s 1.5 # Class -1\n");
    fprintf(gnuplot_pipe,"set style line 3 lc rgb 'green' lt 1 lw 2 # Hyperplane\n");
    fprintf(gnuplot_pipe,"set style line 4 lc rgb 'black' lt 2 lw 2 dashtype 2 # Margins\n");

    // Plotting
    fprintf(gnuplot_pipe, "plot 'class1.dat' with points ls 1 title '+1', \\\n");
    fprintf(gnuplot_pipe, "     'class2.dat' with points ls 2 title '-1', \\\n");
    fprintf(gnuplot_pipe, "     'hyperplane.dat' with lines ls 3 title 'Hyperplane', \\\n");
    fprintf(gnuplot_pipe, "     'margin1.dat' with lines ls 4 title 'Margin +1', \\\n");
    fprintf(gnuplot_pipe, "     'margin2.dat' with lines ls 4 title 'Margin -1'\n");

    fflush(gnuplot_pipe); //Ensure commands are sent immediately
    return 0;
}


// Cleaners
void cleanup_temp_files(){
    remove("class1.dat");
    remove("class2.dat");
    remove("hyperplane.dat");
    remove("margin1.dat");
    remove("margin2.dat");
}

void free_dataset(svm_problem *prob, int free_x_nodes){
    if(prob->x){
        if(free_x_nodes){
            for(int i = 0; i < prob->len; i++) {
                free(prob->x[i]);
            }
        }
        free(prob->x);
    }
    if(prob->y){
        free(prob->y);
    }

}

// Counter how many points lie above, below or in the margins:
// Counts and prints how many points lie above, within, and below the margins, including misclassifications
void counter_points(const svm_model *model, const svm_problem *prob) {
    int pos_above = 0, pos_within = 0, pos_misclassified = 0;
    int neg_below = 0, neg_within = 0, neg_misclassified = 0;
    
    for(int i = 0; i < prob->len; i++) {
        double f1 = 0.0, f2 = 0.0;
        svm_node *x = prob->x[i];
        double y = prob->y[i];
        
        // Extract feature values
        while(x->index != -1) {
            if(x->index == 1) f1 = x->value;
            if(x->index == 2) f2 = x->value;
            x++;
        }
        
        // Compute decision function
        double decision = model->w[0] * f1 + model->w[1] * f2 + model->b;
        
        if(y == 1.0){
            if(decision >= 1.0){
                pos_above++;
            }
            else if(decision >= 0.0 && decision < 1.0){
                pos_within++;
            }
            else{ // decision < 0.0
                pos_misclassified++;
            }
        }
        else if(y == -1.0){
            if(decision <= -1.0){
                neg_below++;
            }
            else if(decision > -1.0 && decision <= 0.0){
                neg_within++;
            }
            else{ // decision > 0.0 that means the point is above the hyper plane and that missclasification ya que label is -1
                neg_misclassified++;
            }
        }
    }
    
    // Print results
    printf("\nClassification Summary:\n");
    printf("Positive Class (+1):\n");
    printf("  - Above Margin: %d\n", pos_above);
    printf("  - Within Margin: %d\n", pos_within);
    printf("  - Misclassified: %d\n", pos_misclassified);
    
    printf("Negative Class (-1):\n");
    printf("  - Below Margin: %d\n", neg_below);
    printf("  - Within Margin: %d\n", neg_within);
    printf("  - Misclassified: %d\n", neg_misclassified);
}
