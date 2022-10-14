#include <iostream>
#include <math.h>
#include <cmath>
using namespace std;
int iterCount = 0;
typedef double(*pointFunc)(double);
double f(double x) {
  return cos(x) * cos(x);
}
double simpson_integral(pointFunc f, double a, double b, int n) {
  const double h = (b-a)/n;
  double k1 = 0, k2 = 0;
  for(int i = 1; i < n; i += 2) {
    k1 += f(a + i*h);
    k2 += f(a + (i+1)*h);
    iterCount ++;
  }
  return h/3*(f(a) + 4*k1 + 2*k2);
}
int main() {
  double a, b, eps;
  double s1, s;
  int n = 1; //начальное число шагов
  a = (double)(-1 / 3);
  b = (double)(1 / 3);
  eps = 1;
  s1 = simpson_integral(f, a, b, n); //первое приближение для интеграла
  do {
    s = s1;     //второе приближение
    n = 2 * n;  //увеличение числа шагов в два раза,
                //т.е. уменьшение значения шага в два раза
    s1 = simpson_integral(f, a, b, n);
  }
  while (fabs(s1 - s) > eps);  //сравнение приближений с заданной точностью
  cout << "\nИнтеграл = " << s1 << endl << iterCount << endl;
}