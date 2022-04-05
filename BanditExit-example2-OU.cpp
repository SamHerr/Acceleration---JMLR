//////////////////////////////////////////////////////////////////////////////////
//
//   title: Exact simulation of diffusion first exit time: algorithm acceleration
//   authors: S. Herrmann and C. Zucca
//   to appear in Journal of Machine Learning Research
//
//   this code concerns the simulation of the first exit time appearing in Example 2
//   on page 13 (Ornstein-Uhlenbeck process): dX=-lambda*X dt+dB
//   We use a rejection sampling and an acceleration by an epsilon-greedy algorithm.
//
//   For the rejection sampling, we refer to the paper: Exact simulation of first exit
//   times for one-dimensional diffusion processes (S. Herrmann and C. Zucca)
//   ESAIM Math. Model. Numer. Anal 54 (2020), no.3, 811--844
//
//
//  Settings:
//  - starting value of the diffusion: xstart (line 500)
//  - exit interval: I=[x_min,x_max] (line 499 and line 501)
//  - drift parameter mu (line  503)
//  - constant kappa (see Herrmann-Zucca 2020, ESAIM M2NA) -- line 502
//  - number of simulations (actions, line 481)
//  - box size (parameter N=MaxNb_slices, line 482)
//  - parameter epsilon (espilon-greedy algorithm, line 483)
//
//  Outcomes:
//  - Computational times: in file "comp-times.txt" (line 564)
//  - Choice of the arm: in file "choice-arms.txt" (line 570)
//  - Exit times: in file "exit-times.txt" (line 582)
//  - Exit location: in file "exit-location.txt" (line 576)
//
//////////////////////////////////////////////////////////////////////////////////
//
//
#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <fstream>
#include <random>
#include <iomanip>
#include <time.h>
#include <chrono>

using namespace std;
//  random number generator
unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::mersenne_twister_engine<uint_fast32_t, 32, 624, 397, 31, 0x9908b0df, 11, 0xffffffff, 7,0x9d2c5680, 15, 0xefc60000, 18, 1812433253> generator (seed);
//std::default_random_engine generator (seed);
std::normal_distribution<double> distribution (0.0,1.0);
std::uniform_real_distribution<double> uniform (0.0,1.0);
//////////////////////////////////////////////////////////////////////////////////
//
//        definition of the drift and related functions /// for the drift 2+sin(x)
//
//////////////////////////////////////////////////////////////////////////////////
double drift(double x,double mu0)
{
    return -mu0*x;
}
double gammafunc(double x,double mu0)
{
    return (pow(mu0*x,2)-mu0)/2;
}
double betafunc(double x, double mu0)
{
    return exp(-mu0*pow(x,2)/2.);
}
double betaplus(double x_min, double x_max, double mu0)
{
    if (x_min*x_max<=0)
    {
        return 1;
    }
    else
    {
        return fmax(betafunc(x_min,mu0),betafunc(x_max,mu0));
    }
}
double gammaplus(double x_min, double x_max, double mu0)
{
    return fmax(gammafunc(x_max,mu0),gammafunc(x_min,mu0));
}
double gammaminus(double x_min, double x_max, double mu0)
{
    if (x_min*x_max<=0)
    {
        return -mu0/2.;
    }
    else
    {
        return fmin(0,fmin(gammafunc(x_max,mu0),gammafunc(x_min,mu0)));
    }
}
double gammazero(double x_min, double x_max, double mu0)
{
    return gammaplus(x_min,x_max,mu0)-gammaminus(x_min,x_max,mu0);
}
//////////////////////////////////////////////////////////////////////////////////
//
//        definition of local functions used in the conddistr procedure
//
//////////////////////////////////////////////////////////////////////////////////
double termserieslong(int a, int n, double x)
{
    if (a==1)
    {
        return (2*n+1)*exp(-2*n*(n+1)/x);
    }
    else
    {
        return (2*n+1)*exp(-n*(n+1)*pow(M_PI,2)*x/2);
    }
}
double refdensityLarge(double x)
{
    return (M_PI/4)*sin((M_PI/2)*(x+1));
}
double termSeriesLarge(int n,double x, double x0, double time)
{
    return exp(-pow(n,2)*pow(M_PI,2)*time/8)*sin(n*(M_PI/2)*(x+1))*sin(n*(M_PI/2)*(x0+1));
}
double ReminderLarge(int n,double time)
{
    return sqrt(2/M_PI/time)*(1-erf(n*M_PI*sqrt(time)/2/sqrt(2)));
}
double refdensitySmall(double x, double time)
{
    return 1/sqrt(2*M_PI*time)*exp(-pow(x,2)/2/time);
}
double termSeriesSmall(int n, double x0, double x, double time)
{
    return refdensitySmall(x0-x-4*n,time)*(abs(x)<1)-refdensitySmall(x0+x-2-4*n,time)*(abs(x)<1);
}
double ReminderSmall(int n, double time)
{
    return (1-erf((4*n-2)/sqrt(2*time)))/4;
}
//////////////////////////////////////////////////////////////////////////////////
//
//        ALGORITHM BROWNIAN_EXIT_SYMM
//
//////////////////////////////////////////////////////////////////////////////////
void BrExitSy(double t0, double& Tvalue)// simulation of the Brownian exit time from the interval [-1,1] with the starting point 0,
//  this algorithm uses a splitting between small times (inverse gaussian method) and large times (exponential distribution method)
//    // t0 represents the threshold which permits to distinguish small and large times
//    // usual value for t0=0.5
{
    double kappainv=M_PI*erf(sqrt((double)1/(2*t0)))*exp(t0*pow(M_PI,2)/8)/2;
    bool test=true;
    int choice=0;
    double X=0;
    while (test)
    {
        double Lower=0;
        double Upper=1;
        int Index=0;
        X=(double)1/(pow(distribution(generator),2));
        if (X<t0)
        {
            choice=1;
        }
        else
        {
            X=t0+(-8/(pow(M_PI,2)))*log(uniform(generator));
            choice=2;
        }
        double W=uniform(generator)*(1+(kappainv-1)*(choice-1));
        while  ((W<Upper) && (test))
        {
            Index+=2;
            Lower=Upper-termserieslong(choice,Index-1,X);
            Upper=Lower+termserieslong(choice,Index,X);
            test= (W>=Lower) ;
        }
    }
    Tvalue=X;
}
//////////////////////////////////////////////////////////////////////////////////
//
//        ALGORITHM BROWNIAN_EXIT_ASYM
//
//////////////////////////////////////////////////////////////////////////////////
void BrExitAsy(double xmin, double xmax, double& Tval, double& Xval)
{
    bool test=true;
    double top=xmax;
    double bottom=xmin;
    double exitime;
    double width;
    int position;
    Tval=0;
    Xval=0;
    while (test)
    {
        if (top<=bottom)
        {
            width=top;
            position=2;
        }
        else
        {
            width=bottom;
            position=1;
        }
        BrExitSy(0.5,exitime);
        Tval+=exitime*pow(width,2);
        if (2*uniform(generator)<1)
        {
            if (position==1)
            {
                Xval=Xval-width;
            }
            else
            {
                Xval=Xval+width;
            }
            test=false;
        }
        else
        {
            if (position==1)
            {
                Xval=Xval+width;
                top=top-width;
                bottom=2*bottom;
            }
            else
            {
                Xval=Xval-width;
                top=2*top;
                bottom=bottom-width;
            }
        }
        if ((top==0) || (bottom==0))
            test=false;
    }
    if (abs(Xval+xmin)<pow(10,-15))
    {
        Xval=-xmin;
    }
    else
    {
        Xval=xmax;
    }
}
//////////////////////////////////////////////////////////////////////////////////
//
//        ALGORITHM CONDITIONAL DISTRIBUTION
//
//////////////////////////////////////////////////////////////////////////////////
void conddistr(double time, double xstart, double& Xval)
{
    Xval=0;
    double Candidate=0;
    double Unif=0;
    double Interm=0;
    double Approx=0;
    int Index=0;
    double t0=0.7;
    int Ksmall=3+floor(sqrt(time)/4);
    int n0=floor(2*sqrt(2)/M_PI/sqrt(time))+1;
    double Klarge=(4/M_PI)*sin(M_PI*(xstart+1)/2)*(8*n0*exp(-pow(n0,2)*pow(M_PI,2)*time/8)/pow(M_PI,2)/time+pow(n0,3)*exp(-pow(M_PI,2)*time/8));
    if (time>t0)
    {
        bool test=true;
        while (test)
        {
            Index=1;
            Approx=0;
            Candidate=2*acos(1-2*uniform(generator))/M_PI-1;
            Unif=uniform(generator);
            Interm=Unif*Klarge*refdensityLarge(Candidate);
            Approx=Approx+termSeriesLarge(Index,Candidate,xstart,time);
            while (abs(Interm-Approx)<ReminderLarge(Index,time))
            {
                Index++;
                Approx+=termSeriesLarge(Index,Candidate,xstart,time);
            }
            test=Interm>Approx;
        }
        Xval=Candidate;
    }
    else
    {
        bool test=true;
        while (test)
        {
            Index=1;
            Approx=0;
            Candidate=distribution(generator)*sqrt(time)+xstart;
            Unif=uniform(generator);
            Interm=Unif*Ksmall*refdensitySmall(Candidate-xstart,time);
            Approx+=termSeriesSmall(Index-1,xstart,Candidate,time)+termSeriesSmall(-Index,xstart,Candidate,time)+termSeriesSmall(Index,xstart,Candidate,time);
            while (abs(Interm-Approx)<ReminderSmall(Index,time))
            {
                Index++;
                Approx+=termSeriesSmall(-Index,xstart,Candidate,time)+termSeriesSmall(Index,xstart,Candidate,time);
            }
            test=Interm>Approx;
        }
        Xval=Candidate;
    }

}
//////////////////////////////////////////////////////////////////////////////////
//
//        ALGORITHM BOXEXIT the size of the Box is [x_min,x_max]*kappa
//
//////////////////////////////////////////////////////////////////////////////////
void BoxExit(double x_start, double x_min, double x_max, double kappa, double mu0, double& T_value, double& X_value)
{
    T_value=0;
    X_value=0;
    double X_interm2=0;
    double X_interm=0;
    double T_interm=0;
    double K_interm=0;
    bool test0=true;
    while (test0)
    {
        X_interm=x_start;
        K_interm=kappa;
        T_value=0;
        bool test1=true;
        while (test1)
        {
            double Erand=-log(uniform(generator))/gammazero(x_min,x_max,mu0);
            double Urand=uniform(generator);
            double Vrand=uniform(generator);
            double Wrand=uniform(generator);
            BrExitAsy(X_interm-x_min,x_max-X_interm,T_interm,X_interm2);
            if ((T_interm<=Erand) && (T_interm<=K_interm))
            {
                X_interm+=X_interm2;
                if ((betaplus(x_min,x_max,mu0)*Urand<betafunc(X_interm,mu0)) && (log(Wrand)<(gammaminus(x_min,x_max,mu0)*(K_interm-T_interm))))
                {
                    test1=false;
                    test0=false;
                    X_value=X_interm;
                    T_value+=T_interm;
                }
                else
                {
                    test1=false;
                }
            }
            if ((K_interm<=Erand) && (T_interm>K_interm))
            {
                conddistr(4*K_interm/pow(x_max-x_min,2),(2*X_interm-x_min-x_max)/(x_max-x_min),X_interm2);
                X_interm2=(x_min+x_max)/2+(x_max-x_min)*X_interm2/2;
                if (betaplus(x_min,x_max,mu0)*Urand<betafunc(X_interm2,mu0))
                {
                    test1=false;
                    test0=false;
                    X_value=X_interm2;
                    T_value+=K_interm;
                }
                else
                {
                    test1=false;
                }
            }
            if ((K_interm>Erand) && (T_interm>Erand))
            {
                conddistr(4*Erand/pow(x_max-x_min,2),(2*X_interm-x_min-x_max)/(x_max-x_min),X_interm2);
                X_interm2=(x_min+x_max)/2+(x_max-x_min)*X_interm2/2;
                if (gammazero(x_min,x_max,mu0)*Vrand>gammafunc(X_interm2,mu0)-gammaminus(x_min,x_max,mu0))
                {
                    X_interm=X_interm2;
                    T_value+=Erand;
                    K_interm=K_interm-Erand;
                }
                else
                {
                    test1=false;
                }
            }
        }
    }
}
////////////////////////////////////////////////////////////////////////////////
//
//        DIFF EXIT
//
////////////////////////////////////////////////////////////////////////////////
//
//
//    index function
int indexfunc(double x_start, double x_min, double x_max, int Nb_slices)
{
    double delta=(x_max-x_min)/(double)Nb_slices;
    if (x_start<=x_min+delta/2.)
    {
        return 1;
    }
    else
    {
        if (x_start>=x_max-delta/2.)
        {
            return Nb_slices-1;
        }
        else
        {
            return (int)floor((x_start-x_min)/delta-0.5)+1;
        }
    }


}
double localmin(double x_min, double x_max, double x_start, int Nb_slices)
{
    double delta=(x_max-x_min)/(double)Nb_slices;
    return x_min+(indexfunc(x_start,x_min,x_max,Nb_slices)-1)*delta;
}
double localmax(double x_min, double x_max, double x_start, int Nb_slices)
{
    double delta=(x_max-x_min)/(double)Nb_slices;
    return x_min+(indexfunc(x_start,x_min,x_max,Nb_slices)+1)*delta;
}
//
//
//
//
//
void DiffExit(double x_start, double x_min, double x_max, double kappa, double mu0, int Nb_slices, double& T_value, double& X_value, int& Counter)
{
    double X_interm=x_start;
    double X_interm2=0;
    double T_interm=0;
    Counter=0;
    T_value=0;
    bool test=((x_start>x_min+pow(10,-15)) && (x_start<x_max-pow(10,-15)));
    while (test)
    {
        BoxExit(X_interm, localmin(x_min,x_max,X_interm,Nb_slices),localmax(x_min,x_max,X_interm,Nb_slices),kappa,mu0,T_interm,X_interm2);
        T_value+=T_interm;
        X_interm=X_interm2;
        Counter+=1;
        test=((X_interm>x_min+pow(10,-15)) && (X_interm<x_max-pow(10,-15)));
    }
    X_value=X_interm;
}
void DiffExitwithoutslices(double x_start, double x_min, double x_max, double kappa, double mu0, double& T_value, double& X_value, int& Counter)
{
    double X_interm=x_start;
    double X_interm2=0;
    double T_interm=0;
    Counter=0;
    T_value=0;
    bool test=((x_start>x_min+pow(10,-15)) && (x_start<x_max-pow(10,-15)));
    while (test)
    {
        BoxExit(X_interm, x_min,x_max,kappa,mu0,T_interm,X_interm2);
        T_value+=T_interm;
        X_interm=X_interm2;
        Counter+=1;
        test=((X_interm>x_min+pow(10,-15)) && (X_interm<x_max-pow(10,-15)));
    }
    X_value=X_interm;
}
//////////////////////////////////////////////////////////////////////////////////
//
//        SIMULATION AND NUMERICS
//
//////////////////////////////////////////////////////////////////////////////////
int main()
{
//////////////////////////////////////////////////////////////////////////////////
//
//
//
//////////////////////////////////////////////////////////////////////////////////
//
//////////////////////////////////////////////////////////////////////////////////
//
//
//
//////////////////////////////////////////////////////////////////////////////////
//
    clock_t start, finish;
    // parameters of the Bandit algorithm
    int actions=10000;
    int MaxNb_slices=20;
    double epsilon=0.1;
    // outcome and local arrays
    int arms[actions+1];
    double times[actions];
    double weight[MaxNb_slices];
    double cumweight[MaxNb_slices];
    double muhat[MaxNb_slices];
    double sojourn[MaxNb_slices];
    double averageTval[MaxNb_slices];
    int min_elt;
    double min_val;
    // outcome variables for the DiffExit function
    double Tval[actions];
    double Xval[actions];
    int Counter;
    // parameters for the DiffExit function
    double x_min=0;
    double x_start=3;
    double x_max=7;
    double kappa=0.5;
    double mu0=1;
    //
    double coin;
    //
    //  Initialisation
    //
    arms[0]=0;
    coin=uniform(generator);
    for (int i=0; i<MaxNb_slices;i++)
    {
        cumweight[i]=(i+1)/((double) MaxNb_slices);
        if (coin>cumweight[i])
        {
            arms[0]+=1;
        }
        muhat[i]=0;
        sojourn[i]=0;
        averageTval[i]=0;
    }
    //
    //  LOOPS
    //
    for (int j=0; j<actions;j++)
    {
        arms[j+1]=0;
        start=clock();
        DiffExit(x_start,x_min,x_max,kappa,mu0,arms[j]+2,Tval[j],Xval[j],Counter);
        finish = clock();
        times[j]=finish-start;
        //
        //  update
        //
        muhat[arms[j]]=(sojourn[arms[j]]*muhat[arms[j]]+times[j])/ ((double) sojourn[arms[j]] +1.);
        sojourn[arms[j]]+=1;
        //
        //     looking for the argmin
        //
        min_elt=0;
        min_val=muhat[0];
        for (int i=0; i<MaxNb_slices;i++)
        {
            if (muhat[i]<min_val)
            {
                min_val=muhat[i];
                min_elt=i;
            }
        }
        //
        //
        //
        coin=uniform(generator);
        for (int i=0; i<MaxNb_slices;i++)
        {
            cumweight[i]=epsilon*(i+1)/((double) MaxNb_slices)+(1-epsilon)*(i>=min_elt);
            if (coin>cumweight[i])
            {
                arms[j+1]+=1;
            }
        }
    }
    const char separateur(' ');
    ofstream sortie1("comp-times.txt",ios::out);
    for(int i=0;i<actions;i++)
        {
            sortie1 << setprecision(8) << times[i] << separateur;
        }
    sortie1.close();
    ofstream sortie2("choice-arms.txt",ios::out);
    for(int i=0;i<actions;i++)
        {
            sortie2 << setprecision(8) << arms[i] << separateur;
        }
    sortie2.close();
    ofstream sortie3("exit-location.txt",ios::out);
    for(int i=0;i<actions;i++)
        {
            sortie3 << setprecision(8) << Xval[i] << separateur;
        }
    sortie3.close();
    ofstream sortie4("exit-times.txt",ios::out);
    for(int i=0;i<actions;i++)
        {
            sortie4 << setprecision(8) << Tval[i] << separateur;
        }
    sortie4.close();

//////////////////////////////////////////////////////////////////////////////////
//
//
//
//////////////////////////////////////////////////////////////////////////////////
    return 0;
}
