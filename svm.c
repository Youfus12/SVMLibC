#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include "svm.h"

// Main Function
int main(int argc, char *argv[])
{

    const char *filename = "datasets/irisExt.csv";  // Path of the data set
    int feature1, feature2;                         // Variables to store the Indices of the two features selected by the user for svm training and plotting
    int selected_features[2];                       // An array to hold the two selected feature indices for easier access
    char continue_choice;                           // Store the decision of the user to continue or exit program


    // i will use a while loop to train and plot the model multiple times with different features without needing to restart the program

    while(1){
        srand(42); // to fix the seed to have the same random value for every iteration.
        svm_problem full_prob = {0}, train_prob = {0}, test_prob = {0};
        scaling_params params;      // Stores the mean and standard deviation for feature scaling
        svm_parameters param;       // Stores the paraeters of the svm model
        svm_model *model = NULL;    // Pointer to the trained SVM model, which will store the weights and bias after training
        double accuracy;            

        // 1) Prompt user to select two features
        printf("\nSelect two features for SVM training and plotting.\n");
        printf("Available features:\n");
        printf("1. Sepal Length (cm)\n2. Sepal Width (cm)\n3. Petal Length (cm)\n4. Petal Width (cm)\n");
        printf("Enter two distinct feature indices (e.g., '1 3'): ");
        if (scanf("%d %d", &feature1, &feature2) != 2) { // will execute and store the values in  feature1 and 2
            fprintf(stderr, "Invalid input. Exiting.\n");
            break;
        }

        // Validate feature indices
        if (feature1 < 1 || feature1 > MAX_FEATURES || feature2 < 1 || feature2 > MAX_FEATURES || feature1 == feature2) {
            fprintf(stderr, "Invalid feature indices. Please enter two distinct integers between 1 and %d.\n", MAX_FEATURES);
            // Clear input buffer
            while ((continue_choice = getchar()) != '\n' && continue_choice != EOF);
            continue;
        }

        selected_features[0] = feature1;
        selected_features[1] = feature2;

        // 2) Load dataset with selected features
        if (load_dataset(filename, &full_prob, MAX_ROWS, selected_features) <= 0) {
            fprintf(stderr, "Failed to load dataset.\n");
            break;
        }

        // 3) Split dataset
        split_dataset(&full_prob, &train_prob, &test_prob, 0.2);

        // 4) Feature Scaling
        compute_scalingParam(&train_prob, &params, 2); // selected_features size is 2, Calculate the mean and the std for each two features, will be stored in param
        // Apply the scaling: each feature value will be scaled 
        apply_scaling(&train_prob, &params);
        apply_scaling(&test_prob, &params);
    
            // Print scaled statistics
        printf("\nScaled Training Data Statistics:\n");
        print_scaled_stats(&params);
        printf("\nScaled Testing Data Statistics:\n");
        print_scaled_stats(&params);
        // Now each feature is approximated to zero mean and unit variance.
 
        // 5) Set parameters
        param.C = 58.0;       // Regularization parameter
        param.eps = 0;    // Not usced in this implementation
        param.eta = 0.01;    // Learning rate
        param.max_iter = 10000000;

        // 6) Train model
        model = svm_train(&train_prob, &param);
        if (!model) {
            fprintf(stderr, "Training failed.\n");
            // Proceed to cleanup
            free_dataset(&full_prob, 1);
            free_dataset(&train_prob, 0);
            free_dataset(&test_prob, 0);
            continue;
        }
        printf("Model trained.\n");

        // Print model weights and bias
        printf("\nModel Weights:\n");
        for(int d = 0; d < 2; d++) {
            printf("w for Feature %d = %.4f\n", selected_features[d], model->w[d]);
        }
        printf("Bias (b) = %.4f\n", model->b);

        // 7) Evaluate
        accuracy = calculate_accuracy(model, &test_prob);
        printf("Test Accuracy: %.2f%%\n", accuracy);

        // 8) Open gnuplot pipe with correct flag
        /*
             Establishes a communication channel between the C program and Gnuplot for real time plotting
            popen : will create a pipe to invoke the shell
            -presist flag: keeps the plot window open even after the program exits(so we can compare to other configurations)
            gnuplot_pipe is a file pointer that can be used to send commands directly to Gnuplot for plotting
        */
    

        FILE *gnuplot_pipe = popen("gnuplot -persist", "w");
        if (!gnuplot_pipe) {
            perror("Failed to open gnuplot");
            svm_free_model(model);
            free_dataset(&full_prob, 1);
            free_dataset(&train_prob, 0);
            free_dataset(&test_prob, 0);
            return EXIT_FAILURE;
        }

        // 9) Write plot data (including margins): If write_plot_data fails (returns non zero), an error message is displayed, and frees the allocated memory

        if (write_plot_data(&train_prob, model) != 0) {
            fprintf(stderr, "Failed to write plot data. Skipping plot.\n");
            pclose(gnuplot_pipe);
            svm_free_model(model);
            free_dataset(&full_prob, 1);
            free_dataset(&train_prob, 0);
            free_dataset(&test_prob, 0);
            continue;
        }

        // Count points relative to margins
        counter_points(model, &train_prob); // will count how many points are in(in margin) or above(upper margin) or lower (lower margin)

        // Plot data (including margins)
        plot_data(gnuplot_pipe, selected_features[0], selected_features[1]); 
        //The plot_data function sends appropriate commands to Gnuplot to read these files and generate the visual plot
        printf("Plot generated for features %d and %d.\n", selected_features[0], selected_features[1]);

        // 10) Cleanup for this iteration
        pclose(gnuplot_pipe);
        svm_free_model(model);
        free_dataset(&full_prob, 1);      // Free x_nodes
        free_dataset(&train_prob, 0);     // Do not free x_nodes (cause train and test have the same x_nodes, and its free)
        free_dataset(&test_prob, 0);      // Do not free x_nodes
        free(params.mean);
        free(params.std);

        //Remove temporary plot data files
        cleanup_temp_files();

        // 11) Prompt user to continue or exit
        printf("\nDo you want to train and plot with another pair of features? (y/n): ");
        fflush(stdout); // Ensure the prompt is displayed immediately
        scanf(" %c", &continue_choice);
        if(continue_choice == 'n' || continue_choice == 'N'){
            printf("Exiting the program.\n");
            break;
        }
    }

    return 0;
}
