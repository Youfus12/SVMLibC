#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <SDL2/SDL.h>
#include <math.h>

// Define the number of data points
#define NUM_POINTS 100

// Define the window dimensions
#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 480

// Define the size of the points
#define POINT_SIZE 10

// Define the margin width (for visualization purposes)
#define MARGIN_WIDTH 40

// Define the color structure
typedef struct {
    Uint8 r, g, b;
} Color;

// Function to generate data points
void generate_data(int x[], int y[], Color colors[]) {
    srand(time(NULL));

    // Create red points inside a circle (class +1)
    for (int i = 0; i < NUM_POINTS / 2; i++) {
        x[i] = rand() % (WINDOW_WIDTH / 2);  // Points within half the width (left side)
        y[i] = rand() % (WINDOW_HEIGHT / 2); // Points within half the height (top side)
        colors[i] = (Color){255, 0, 0};  // Red for class +1 (inside)
    }

    // Create blue points outside the circle (class -1)
    for (int i = NUM_POINTS / 2; i < NUM_POINTS; i++) {
        x[i] = rand() % (WINDOW_WIDTH / 2) + (WINDOW_WIDTH / 2);  // Points on right side
        y[i] = rand() % (WINDOW_HEIGHT / 2) + (WINDOW_HEIGHT / 2); // Points on bottom side
        colors[i] = (Color){0, 0, 255};  // Blue for class -1 (outside)
    }
}

// Function to calculate the dot product
double dot_product(double *a, double *b, int n) {
    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// SVM model with weights and bias
typedef struct {
    double w[2];  // Weight vector for 2D data
    double b;     // Bias term
} SVMModel;

// Function to update the model using simple gradient descent
void update_model(SVMModel *model, int *x, int *y, int n, double learning_rate) {
    double dw[2] = {0.0, 0.0};
    double db = 0.0;

    for (int i = 0; i < n; i++) {
        double margin = y[i] * (dot_product(model->w, (double[]){x[i], y[i]}, 2) + model->b);
        if (margin < 1) {
            dw[0] += -y[i] * x[i];
            dw[1] += -y[i] * y[i];
            db += -y[i];
        }
    }

    // Regularization for weights
    dw[0] += model->w[0];
    dw[1] += model->w[1];

    // Apply gradient descent updates
    model->w[0] -= learning_rate * dw[0];
    model->w[1] -= learning_rate * dw[1];
    model->b -= learning_rate * db;
}

// Function to train the SVM
void train_svm(SVMModel *model, int *x, int *y, int n, double learning_rate, int max_iter) {
    model->w[0] = 0.0;
    model->w[1] = 0.0;
    model->b = 0.0;

    for (int i = 0; i < max_iter; i++) {
        update_model(model, x, y, n, learning_rate);
    }
}

// Function to draw dashed lines
void draw_dashed_line(SDL_Renderer *renderer, int x1, int y1, int x2, int y2, int dash_length) {
    int dx = x2 - x1;
    int dy = y2 - y1;
    int dash_count = (int)(sqrt(dx * dx + dy * dy) / dash_length);
    
    for (int i = 0; i < dash_count; i++) {
        int start_x = x1 + i * dx / dash_count;
        int start_y = y1 + i * dy / dash_count;
        int end_x = x1 + (i + 1) * dx / dash_count;
        int end_y = y1 + (i + 1) * dy / dash_count;

        if (i % 2 == 0) { // Draw only on alternate segments
            SDL_RenderDrawLine(renderer, start_x, start_y, end_x, end_y);
        }
    }
}

// Function to visualize the SVM with margin lines (dashed black)
void visualize_svm(int *x, int *y, Color *colors, int n, SVMModel *model) {
    // SDL setup
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *window = SDL_CreateWindow("SVM Visualization", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);  // White background
    SDL_RenderClear(renderer);

    // Draw data points
    for (int i = 0; i < n; i++) {
        SDL_Rect rect = {x[i] - POINT_SIZE / 2, y[i] - POINT_SIZE / 2, POINT_SIZE, POINT_SIZE};
        SDL_SetRenderDrawColor(renderer, colors[i].r, colors[i].g, colors[i].b, 255);
        SDL_RenderFillRect(renderer, &rect);  // Draw larger points as rectangles
    }

    // Draw the decision boundary (w1 * x1 + w2 * x2 + b = 0 -> line equation)
    SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);  // Green for decision boundary
    int x1 = -WINDOW_WIDTH / 2, x2 = WINDOW_WIDTH / 2;
    int y1 = -(model->w[0] * x1 + model->b) / model->w[1];
    int y2 = -(model->w[0] * x2 + model->b) / model->w[1];
    SDL_RenderDrawLine(renderer, x1 + WINDOW_WIDTH / 2, y1 + WINDOW_HEIGHT / 2, x2 + WINDOW_WIDTH / 2, y2 + WINDOW_HEIGHT / 2);

    // Draw the upper margin line (parallel to the decision boundary)
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);  // Black for margin
    int margin_y1 = -(model->w[0] * x1 + model->b + MARGIN_WIDTH) / model->w[1];
    int margin_y2 = -(model->w[0] * x2 + model->b + MARGIN_WIDTH) / model->w[1];
    draw_dashed_line(renderer, x1 + WINDOW_WIDTH / 2, margin_y1 + WINDOW_HEIGHT / 2, x2 + WINDOW_WIDTH / 2, margin_y2 + WINDOW_HEIGHT / 2, 10);

    // Draw the lower margin line (parallel to the decision boundary)
    int margin_y1_lower = -(model->w[0] * x1 + model->b - MARGIN_WIDTH) / model->w[1];
    int margin_y2_lower = -(model->w[0] * x2 + model->b - MARGIN_WIDTH) / model->w[1];
    draw_dashed_line(renderer, x1 + WINDOW_WIDTH / 2, margin_y1_lower + WINDOW_HEIGHT / 2, x2 + WINDOW_WIDTH / 2, margin_y2_lower + WINDOW_HEIGHT / 2, 10);

    // Show the result on screen
    SDL_RenderPresent(renderer);
    SDL_Delay(5000);  // Show for 5 seconds

    // Cleanup SDL
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

// SDL requires `int SDL_main(int argc, char *argv[])` instead of `int main()`
int SDL_main(int argc, char *argv[]) {
    // Data generation
    int x[NUM_POINTS], y[NUM_POINTS];
    Color colors[NUM_POINTS];
    generate_data(x, y, colors);

    // Prepare SVM model
    SVMModel model;
    train_svm(&model, x, y, NUM_POINTS, 0.01, 1000);  // Train the model with learning rate = 0.01 and 1000 iterations

    // Visualize the result
    visualize_svm(x, y, colors, NUM_POINTS, &model);

    return 0;
}




