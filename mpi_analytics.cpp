#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

void readCSV(string filename, vector<double>& value1)
{
    ifstream file(filename);

    if(!file)
    {
        cout << "Cannot open file\n";
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    string line,a,b;

    getline(file,line);

    while(getline(file,line))
    {
        stringstream ss(line);

        getline(ss,a,',');

        getline(ss,b);

        value1.push_back(stod(a));
    }

    file.close();
}

int main(int argc, char* argv[])
{
    int rank,size;

    MPI_Init(&argc,&argv);

    double start=MPI_Wtime();

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    vector<double> data;

    if(rank==0)
    {
        readCSV("dataset/small.csv",data);

        cout<<"Records : "<<data.size()<<endl;
    }

    int n;

    if(rank==0)
        n=data.size();

    MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);

        if (n == 0)
        {
            if(rank == 0)
                cout << "Dataset is empty!" << endl;

            MPI_Finalize();
            return 0;
        }
        if(rank==0)
        {
            if(n % size != 0)
            {
                cout << "Dataset size is not divisible by number of processes." << endl;
            }
        }

    int localSize=n/size;

    vector<double> local(localSize);

    MPI_Scatter(
        data.data(),
        localSize,
        MPI_DOUBLE,
        local.data(),
        localSize,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    double localSum = 0;

    double localMin = local[0];
    double localMax = local[0];

    for(double v : local)
    {
        localSum += v;

        if(v < localMin)
            localMin = v;

        if(v > localMax)
            localMax = v;
    }

    double globalSum = 0;
    double globalMin = 0;
    double globalMax = 0;

    MPI_Reduce(
        &localSum,
        &globalSum,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );
    MPI_Reduce(
        &localMin,
        &globalMin,
        1,
        MPI_DOUBLE,
        MPI_MIN,
        0,
        MPI_COMM_WORLD
    );

    MPI_Reduce(
            &localMax,
            &globalMax,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
        );
        
    double end=MPI_Wtime();
    
    if(rank==0)
    {
        double mean=globalSum/n;

        cout << "\n===== MPI Analytics Result =====\n";
        cout << "Records           : " << n << endl;
        cout << "Processes Used    : " << size << endl;
        cout << "Mean              : " << mean << endl;
        cout << "Minimum           : " << globalMin << endl;
        cout << "Maximum           : " << globalMax << endl;
        cout << "Execution Time    : "
            << end - start
            << " seconds" << endl;
    }   

    if(rank==0)
    {
        cout<<"Execution Time = "
            <<end-start
            <<" seconds"<<endl;
    }
    MPI_Finalize();
    return 0;
}
