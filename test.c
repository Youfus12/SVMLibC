#include <stdio.h>
#include <stdlib.h>


#ifndef MIN
#define MIN(x,y) ((x)<(y) ? (x):(y))
#endif

#ifndef MAX
#define MAX(x,y) ((x)>(y) ? (x):(y))
#endif

static inline void swap_int(int *x, int *y)
{
    int t = *x;
    *x = *y;
    *y = t;
}

static inline void swap_double(double *x, double *y)
{
    double t = *x;
    *x = *y;
    *y = t;
}

int main(){
    int x = 5;
    int y = 4;
    
    swap_int(&x,&y);

    printf("x :%d",x);

    printf("\n");
    return 0;
}

