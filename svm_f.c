svm_f
int load_dataset(const char filename, svm_problemprob, int max_rows)
{
    FILE file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        return -1;
    }

    prob->l = 0;
    prob->y = (double)calloc(max_rows, sizeof(double));
    prob->x = (svm_node *)calloc(max_rows, sizeof(svm_node));

    char line[256];
    // Attempt to read header line
    if (fgets(line, sizeof(line), file)) {

    }

    while (fgets(line, sizeof(line), file) && prob->l < max_rows) {
        double f1, f2, f3, f4;
        int target;

        // Expecting 5 fields
        int fields = sscanf(line, "%lf,%lf,%lf,%lf,%d", &f1, &f2, &f3, &f4, &target);
        if (fields != 5) {
            fprintf(stderr, "Skipping invalid line: %s\n", line);
            continue;
        }

        if (target != 0 && target != 1) {
            // skip lines that aren't binary 0/1
            continue;
        }

        double label = (target == 0) ? 1.0 : -1.0;

        // Allocate feature array for 4 features plus a terminator
        svm_node x_node = (svm_node)malloc((MAX_FEATURES + 1) * sizeof(svm_node));
        x_node[0].index = 1; x_node[0].value = f1;
        x_node[1].index = 2; x_node[1].value = f2;
        x_node[2].index = 3; x_node[2].value = f3;
        x_node[3].index = 4; x_node[3].value = f4;
        x_node[4].index = -1;  // terminator

        prob->y[prob->l] = label;
        prob->x[prob->l] = x_node;
        prob->l++;
    }
    fclose(file);
    return prob->l;
}