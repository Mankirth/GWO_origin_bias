/// An example code to test the problems of the
///     Congress on Evolutionary Computation 2020
///     Competition on Real-World Constrained Optimization
/// Author: Vladimir Stanovov (vladimirstanovov@yandex.ru)
///     Reshetnev Siberian State University of Science and Technology
///     Krasnoyarsk, Russian Federation
/// Last change: 21/04/2025

#include <time.h>
#include <fstream>
#include <random>
#include "cec2020rwc.cpp"

unsigned globalseed = 2025;
std::mt19937 generator_uni_r(globalseed);
std::uniform_real_distribution<double> uni_real(0.0,1.0);
double Random(double minimal, double maximal) {return uni_real(generator_uni_r)*(maximal-minimal)+minimal;}

int main()
{
    for(int fn=1;fn!=2;fn++)
    {
        cout<<"Testing function "<<fn<<endl;
        int func_num = fn;
        int DIM = global_D[func_num-1]; // get problem dimension
        int NumG = global_gn[func_num-1]; // get number of inequality constraints
        int NumH = global_hn[func_num-1]; // get number of equality constraints
        double* lowb = new double[DIM]; // lower boundary
        double* upb = new double[DIM]; // upper boundary
        get_bounds(func_num,lowb,upb);
        double* xval = new double[DIM]; // vector to be evaluated
        double* gval = new double[NumG]; // g values
        double* hval = new double[NumH]; // h values
        double fval[1]; // target function value
        cout.precision(16);
        char buffer[100];
        sprintf(buffer,"test_func_%d.txt",fn);
        ofstream fout(buffer);
        fout.precision(18); // set better precision for comparison with matlab
        for(int testnum=0;testnum!=3;testnum++)
        {
            cout<<fn<<"\t"<<testnum<<endl;
            cout<<"[";
            for(int i=0;i!=DIM;i++)
            {
                xval[i] = Random(lowb[i],upb[i]);
                fout<<xval[i]<<"\t";
                cout<<xval[i];
                if(i != DIM-1)
                    cout<<",\t";
            }
            cout<<"]"<<endl;
            cout<<endl;
            fout<<endl;
            cec20_func(xval,func_num,fval,gval,hval); // evaluate CEC 2020 function
            fout<<fval[0]<<"\t";//<<endl;
            cout<<fval[0]<<endl;
            cout<<endl;
            for(int i=0;i!=NumG;i++)
            {
                cout<<gval[i]<<"\t";
                fout<<gval[i]<<"\t";
            }
            cout<<endl;
            fout<<endl;
            for(int i=0;i!=NumH;i++)
            {
                cout<<hval[i]<<"\t";
                fout<<hval[i]<<"\t";
            }
            cout<<endl;
            fout<<endl;
        }
        delete xval;
        delete gval;
        delete hval;
        delete lowb;
        delete upb;
    }
    return 0;
}
