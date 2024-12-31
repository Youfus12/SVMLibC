#ifndef __SVM_H__
#define __SVM_H__

#include "inst.h"
#include "def.h"

typedef struct jmplbl_t {
    int no;
    char* name;
    int jmppos;
} jmplbl_t;

#define SM_ARGS int chr, int* state, char* name, int* name_len, char** argv, \
        int* argv_len, int* argc, int line, int* out_bin, int* pos
/* syntax parser state machines */
int sm_symbol_name(SM_ARGS);
int sm_space(SM_ARGS);
int sm_newline(SM_ARGS);
int sm_comma(SM_ARGS);
int sm_colon(SM_ARGS);
int sm_semicolon(SM_ARGS);
int sm_eof(SM_ARGS);

int util_stricmp(const char* s1, const char* s2);
int util_isnumeric(const char* text);

void init_registers(void);

int inst_len(const char* name);
int parse_jmplbl(FILE* infile);
int parse_file(FILE* infile, int** output);
int parse_arg(int type_chr, const char* text, int* bin_ptr);
int exec_binary(int* bin, int len, int startpoint);

int* get_register(int reg_no);
int get_rvalue(int type, int val);
void push_callstack(int pos);
int pop_callstack(void);
void push_stack(int val);
int pop_stack(void);

int trans_inst(const char* name, int argc, char* const* argv, int* out_bin);

#endif /* __SVM_H__ */

#include<stdio.h>
#include <stdlib.h>
#include "svm.h"
//svm model train
struct svm_model *train_model(struct svm_problem *prob, struct svm_parameter *param) {
    struct svm_model *model = svm_train(prob, param);
    return model;
}


//predict
double predict(struct svm_model *model, struct svm_node *x) {
    return svm_predict(model, x);
}


//save model
void save_model(struct svm_model *model, const char *filename) {
    if (svm_save_model(filename, model) == 0) {
        printf("Model saved successfully to %s\n", filename);
    } else {
        printf("Error saving the model.\n");
    }
}


//svm model load
struct svm_model *load_model(const char *filename) {
    struct svm_model *model = svm_load_model(filename);
    if (model == NULL) {
        printf("Error loading the model.\n");
    }
    return model;
}


//cross validation
void cross_validation(struct svm_problem *prob, struct svm_parameter *param, int nr_fold) {
    double *target = malloc(sizeof(double) * prob->l);
    svm_cross_validation(prob, param, nr_fold, target);

    // طباعة النتائج
    for (int i = 0; i < prob->l; i++) {
        printf("Sample %d: predicted label = %f\n", i, target[i]);
    }

    free(target);
}


// free model
void free_model(struct svm_model **model) {
    svm_free_and_destroy_model(model);
}


//destroy param
void destroy_param(struct svm_parameter *param) {
    svm_destroy_param(param);
}



#define NUM_SAMPLES 150
#define NUM_FEATURES 4

// Fonction pour charger les données iris depuis un fichier CSV (à implémenter selon votre format)
void load_data(double X[NUM_SAMPLES][NUM_FEATURES], int Y[NUM_SAMPLES]) {
    // Charger les données à partir d'un fichier ou d'une autre source
    // Exemple simplifié avec des données fictives
    for (int i = 0; i < NUM_SAMPLES; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            X[i][j] = (rand() % 100) / 100.0;  // Remplacer par vos données réelles
        }
        Y[i] = rand() % 3;  // Remplacer par vos étiquettes cibles réelles
    }
}

// Fonction pour séparer les données en ensembles d'entraînement et de test (simplement pour illustration)
void split_data(double X[NUM_SAMPLES][NUM_FEATURES], int Y[NUM_SAMPLES], 
                double X_train[NUM_SAMPLES][NUM_FEATURES], int Y_train[NUM_SAMPLES], 
                double X_test[NUM_SAMPLES][NUM_FEATURES], int Y_test[NUM_SAMPLES]) {
    int train_size = (int)(NUM_SAMPLES * 0.67);
    for (int i = 0; i < train_size; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            X_train[i][j] = X[i][j];
        }
        Y_train[i] = Y[i];
    }
    for (int i = train_size; i < NUM_SAMPLES; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            X_test[i - train_size][j] = X[i][j];
        }
        Y_test[i - train_size] = Y[i];
    }
}

int main() {
    // Déclaration des données
    double X[NUM_SAMPLES][NUM_FEATURES];
    int Y[NUM_SAMPLES];
    double X_train[NUM_SAMPLES][NUM_FEATURES], Y_train[NUM_SAMPLES];
    double X_test[NUM_SAMPLES][NUM_FEATURES], Y_test[NUM_SAMPLES];
    
    // Charger les données
    load_data(X, Y);

    // Séparer en données d'entraînement et de test
    split_data(X, Y, X_train, Y_train, X_test, Y_test);

    // Création du problème SVM
    struct svm_problem prob;
    prob.l = NUM_SAMPLES;  // Nombre d'échantillons

    // Allocation de la mémoire pour les instances
    prob.x = (struct svm_node**)malloc(NUM_SAMPLES * sizeof(struct svm_node*));
    for (int i = 0; i < NUM_SAMPLES; i++) {
        prob.x[i] = (struct svm_node*)malloc((NUM_FEATURES + 1) * sizeof(struct svm_node));
        for (int j = 0; j < NUM_FEATURES; j++) {
            prob.x[i][j].index = j + 1;
            prob.x[i][j].value = X[i][j];  // Remplir avec les données d'entraînement
        }
        prob.x[i][NUM_FEATURES].index = -1;  // Fin de la ligne pour chaque échantillon
    }

    // Remplir les labels Y
    prob.y = Y;

    // Définir les paramètres du modèle SVM
    struct svm_parameter param;
    param.svm_type = C_SVC;
    param.kernel_type = LINEAR;
    param.degree = 3;
    param.gamma = 0;  // Pas de kernel RBF, car nous utilisons un kernel linéaire
    param.coef0 = 0;
    param.C = 100;  // C = 100
    param.eps = 0.001;
    param.cache_size = 100;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;

    // Entraîner le modèle SVM
    struct svm_model *model = svm_train(&prob, &param);

    // Tester le modèle sur les données de test (calculer la précision)
    int correct = 0;
    for (int i = 0; i < NUM_SAMPLES - NUM_SAMPLES * 0.67; i++) {
        struct svm_node* test_instance = (struct svm_node*)malloc((NUM_FEATURES + 1) * sizeof(struct svm_node));
        for (int j = 0; j < NUM_FEATURES; j++) {
            test_instance[j].index = j + 1;
            test_instance[j].value = X_test[i][j];
        }
        test_instance[NUM_FEATURES].index = -1;  // Fin de la ligne
        double predicted = svm_predict(model, test_instance);

        if ((int)predicted == Y_test[i]) {
            correct++;
        }
        free(test_instance);
    }

    // Calculer la précision
    printf("Accuracy: %.2f%%\n", (correct / (double)(NUM_SAMPLES * 0.33)) * 100);

    // Libérer la mémoire
    svm_free_and_destroy_model(&model);
    for (int i = 0; i < NUM_SAMPLES; i++) {
        free(prob.x[i]);
    }
    free(prob.x);

    return 0;
}
