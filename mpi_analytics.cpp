#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>

using namespace std;

void readCSV(string filename, vector<double>& value1, vector<double>& value2)
{
    ifstream file(filename);

    if(!file)
    {
        cout << "Cannot open file\n";
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    string line,a,b;

    getline(file,line);   // Skip header

    while(getline(file,line))
    {
        stringstream ss(line);

        getline(ss,a,',');
        getline(ss,b);

        value1.push_back(stod(a));
        value2.push_back(stod(b));
    }

    file.close();
}
int main(int argc, char* argv[])
{
    int rank,size;

    MPI_Init(&argc,&argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<double> value1;
    vector<double> value2;

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
        readCSV(filename, value1, value2);

        cout << "Records : " << value1.size() << endl;
    }
    int n = 0;

    if(rank==0)
        n = static_cast<int>(value1.size());

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

    vector<double> localValue1(localSize);
    vector<double> localValue2(localSize);
        if(localSize == 0)
        {
            MPI_Finalize();
            return 0;
        }
    MPI_Scatter(
        value1.data(),
        localSize,
        MPI_DOUBLE,
        localValue1.data(),
        localSize,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Scatter(
        value2.data(),
        localSize,
        MPI_DOUBLE,
        localValue2.data(),
        localSize,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    double localSum = 0;
    double localMin = localValue1[0];
    double localMax = localValue1[0];

    for(double v : localValue1)
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
    double localNumerator = 0;
    double localDx2 = 0;
    double localDy2 = 0;
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

    double meanValue2 = 0;

    // Calculate local sum of Value2
    double localSum2 = 0;
    for(double v : localValue2)
    {
        localSum2 += v;
    }

    double globalSum2 = 0;

    MPI_Reduce(
        &localSum2,
        &globalSum2,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    if(rank == 0)
    {
        meanValue2 = globalSum2 / n;
    }

    MPI_Bcast(
        &meanValue2,
        1,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    for(int i = 0; i < localSize; i++)
    {
        double dx = localValue1[i] - mean;
        double dy = localValue2[i] - meanValue2;

        localVariance += dx * dx;

        localNumerator += dx * dy;
        localDx2 += dx * dx;
        localDy2 += dy * dy;
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
    double globalNumerator = 0;
    double globalDx2 = 0;
    double globalDy2 = 0;

    MPI_Reduce(
        &localNumerator,
        &globalNumerator,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    MPI_Reduce(
        &localDx2,
        &globalDx2,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    MPI_Reduce(
        &localDy2,
        &globalDy2,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );
    // ---------- Local Histogram ----------
    vector<int> localHistogram(10, 0);

    for(double v : localValue1)
    {
        int bin = static_cast<int>(v / 1000);

        if(bin > 9)
            bin = 9;

        localHistogram[bin]++;
    }
    vector<int> gatheredHistogram;

    if(rank == 0)
    {
        gatheredHistogram.resize(size * 10);
    }

    MPI_Gather(
        localHistogram.data(),
        10,
        MPI_INT,
        gatheredHistogram.data(),
        10,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    double end=MPI_Wtime();
        
    if(rank == 0)
    {
        double variance = globalVariance / n;
        double standardDeviation = sqrt(variance);
        double correlation = globalNumerator / sqrt(globalDx2 * globalDy2);
        vector<int> finalHistogram(10,0);

        for(int p = 0; p < size; p++)
        {
            for(int i = 0; i < 10; i++)
            {
                finalHistogram[i] += gatheredHistogram[p * 10 + i];
            }
        }
    ofstream out("results/mpi_results.csv");

    if(!out)
    {
        cout << "Failed to create results file." << endl;
    }
    else
    {

        out << "Metric,Value\n";
        out << "Dataset Size," << datasetSize << "\n";
        out << "Records," << n << "\n";
        out << "Processes Used," << size << "\n";
        out << "Mean," << mean << "\n";
        out << "Minimum," << globalMin << "\n";
        out << "Maximum," << globalMax << "\n";
        out << "Variance," << variance << "\n";
        out << "Standard Deviation," << standardDeviation << "\n";
        out << "Pearson Correlation," << correlation << "\n";
        out << "Execution Time (s)," << end - start << "\n";
        out << "\nHistogram Bin,Count\n";

        for(int i = 0; i < 10; i++)
        {
            out << i * 1000
                << "-" << (i + 1) * 1000 - 1
                << ","
                << finalHistogram[i]
                << "\n";
        }
        out.close();

        cout << "\n===== MPI Analytics Result =====\n";
        cout << "Records           : " << n << endl;
        cout << "Processes Used    : " << size << endl;
        cout << "Mean              : " << mean << endl;
        cout << "Minimum           : " << globalMin << endl;
        cout << "Maximum           : " << globalMax << endl;
        cout << "Variance          : " << variance << endl;
        cout << "Std Deviation     : " << standardDeviation << endl;
        cout << "Pearson Correlation : " << correlation << endl;
        cout << "\n========== HISTOGRAM ==========\n";

            for(int i = 0; i < 10; i++)
            {
            cout << setw(4) << i * 1000
                << " - "
                << setw(5) << (i + 1) * 1000 - 1
                << " : "
                << finalHistogram[i]
                << endl;
            }
        cout << "Execution Time    : " << end - start << " seconds" << endl;
        cout << "Results saved to results/mpi_results.csv" << endl;
    }
}
       

    MPI_Finalize();
    return 0;
}
