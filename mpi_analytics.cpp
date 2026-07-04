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

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<double> data;

    string filename;

    if (argc < 2)
    {
        if (rank == 0)
            cout << "Usage: mpi_analytics <dataset_size>" << endl;

        MPI_Finalize();
        return 1;
    }
    int datasetSize = stoi(argv[1]);

    if (datasetSize == 1000000)
        filename = "dataset/small.csv";
    else if (datasetSize == 10000000)
        filename = "dataset/medium.csv";
    else if (datasetSize == 100000000)
        filename = "dataset/large.csv";
    else
    {
        if (rank == 0)
            cout << "Invalid dataset size!" << endl;

        MPI_Finalize();
        return 1;
    }

    if (rank == 0)
    {
        readCSV(filename, data);

        cout << "Records : " << data.size() << endl;
    }
    int n = 0;

    if(rank==0)
        n = static_cast<int>(data.size());

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
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    double start = MPI_Wtime();

    int localSize=n/size;

    vector<double> local(localSize);
        if(localSize == 0)
        {
            MPI_Finalize();
            return 0;
        }
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
    //Global Results
    double globalSum = 0;
    double globalMin = 0;
    double globalMax = 0;
    double globalVariance = 0;
    //Reduce Sum
    MPI_Reduce(
        &localSum,
        &globalSum,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );
    //Reduce Minimum
    MPI_Reduce(
        &localMin,
        &globalMin,
        1,
        MPI_DOUBLE,
        MPI_MIN,
        0,
        MPI_COMM_WORLD
    );
    //Reduce Maximum
    MPI_Reduce(
            &localMax,
            &globalMax,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
    );
    //Calculate Mean (Rank 0)
    double mean = 0;

    if(rank == 0)
    {
        mean = globalSum / n;
    }
    //BRoadcast Mean
    MPI_Bcast(
        &mean,
        1,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );
    //Calculate Local Variance
    double localVariance = 0;

    for(double v : local)
    {
        localVariance += (v - mean) * (v - mean);
    }
    //Reduce Variance
    MPI_Reduce(
        &localVariance,
        &globalVariance,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );


    double end=MPI_Wtime();
        
    if(rank == 0)
    {
        double variance = globalVariance / n;
        double standardDeviation = sqrt(variance);

        cout << "\n===== MPI Analytics Result =====\n";
        cout << "Records           : " << n << endl;
        cout << "Processes Used    : " << size << endl;
        cout << "Mean              : " << mean << endl;
        cout << "Minimum           : " << globalMin << endl;
        cout << "Maximum           : " << globalMax << endl;
        cout << "Variance          : " << variance << endl;
        cout << "Std Deviation     : " << standardDeviation << endl;
        cout << "Execution Time    : " << end - start << " seconds" << endl;
    }
            

    MPI_Finalize();
    return 0;
}
