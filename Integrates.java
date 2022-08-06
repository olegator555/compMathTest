import java.util.stream.IntStream;
import static java.lang.Math.sqrt;
import java.util.ArrayList;
import static java.lang.Math.*;

public class Integrates {
    public static void main(String[] args) {
        for(int i=3; i<1000; i++)
        {
            Main_function f = new Main_function();
            ArrayList<Double> simpson_series = new ArrayList<>();
            ArrayList<Double> trapezoid_series = new ArrayList<>();
            simpson_series.add(composite_simpson(10e-6,Functions.T,i, f)-Functions.A);
            trapezoid_series.add(composite_trapezoid(10e-6,Functions.T,i,f)-Functions.A);
            System.out.println(simpson_series);
            System.out.println(trapezoid_series);


        }
    }
    public static double composite_simpson(double a, double b, int n, Main_function f){
        double h = (b-a)/n;
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
        double dx = C * (-cos(2 * x));
        double y = C * (0.5 - 0.5 * cos(2 * x));
        double dy = sin(2 * x) / (1 - cos(2 * x));
        return sqrt((1 + pow(dy, 2)) / (2 * g * y)) * dx;
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