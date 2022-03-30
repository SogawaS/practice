#include<stdio.h>
#include<stdlib.h>
int main(void){


    for(int i=0;i<160;i++){
        if(i%5==0) printf("%d\n",i/5+1);
        if(i%5!=0) printf("%d ",rand()%101);
    }
    return 0;
}