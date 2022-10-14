import java.util.stream.IntStream;
import static java.lang.Math.*;
import java.util.ArrayList;
import static java.lang.Math.*;

public class Integrates {
    int n = 3;
    public static void main(String[] args) {
        Main_function main_function = new Main_function();
        System.out.println(composite_simpson(-1/3, 1/3, 3, main_function));
    }
    public static double composite_simpson(double a, double b, int n, Main_function f){
        double h = (b-a)/n;
        System.out.println(h);
        double k1 = 0, k2 = 0;
        for(int i = 1; i < n; i += 2) {
            k1 += f.getFunction(a + i*h);
            k2 += f.getFunction(a + (i+1)*h);
        }
        return h/3*(f.getFunction(a) + 4*k1 + 2*k2);
    }

    public static double composite_trapezoid(double a, double b, int n, Main_function f){
        final double width = (b-a)/n;
        double result = 0;
        for(int step = 0; step < n; step++) {
            final double x1 = a + step*width;
            final double x2 = a + (step+1)*width;

            result += 0.5*(x2-x1)*(f.getFunction(x1) + f.getFunction(x2));
        }

        return result;
    }
}
class Main_function implements Functions {
    @Override
    public double getFunction(double x) {
        return cos(x)*cos(x);
    }

    @Override
    public double[] approximate_n(int[] n) {
        ArrayList<Double> list = new ArrayList<>();
        for (int elem:n
        ) {
            list.add((elem-A)/elem);
        }
        return list.stream().mapToDouble(i->i).toArray();
    }


}
interface Functions {
     double C = 1.03439984;
     double T = 1.75418438;
     double g = 9.81;
     double A = sqrt(2*C/g)*(T-10e-5);
     int[] n = IntStream.rangeClosed(0,1000).toArray();
     double getFunction(double x);
     double[] approximate_n(int[] n);

}