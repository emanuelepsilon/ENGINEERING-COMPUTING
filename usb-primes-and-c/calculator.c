#include <stdio.h>

int main() {
    double a, b, result;
    char op;
    printf("Enter: number operator number\n");
    scanf("%lf %c %lf", &a, &op, &b);

    switch(op) {
        case '+':
            result = a + b;
            break;
        case '-':
            result = a - b;
            break;
        case '*':
            result = a * b;
            break;
        case '/':
            if (b != 0)
                result = a / b;
            else {
                printf("Error: Division by zero!\n");
                return 1;
            }
            break;
        default:
            printf("Error: Unknown operator '%c'\n", op);
            return 1;
    }

    printf("Result: %.2lf\n", result);
    return 0;
}
