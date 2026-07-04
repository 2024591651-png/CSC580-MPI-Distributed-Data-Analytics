#include <iostream>
#include <fstream>
#include <random>
#include <string>

using namespace std;

// Function to generate a dataset
void generateDataset(string filename, long long size)
{
    cout << "Generating " << filename << "..." << endl;

    ofstream file(filename);

    if (!file)
    {
        cout << "Failed to create " << filename << endl;
        return;
    }

    // CSV Header
    file << "Value1,Value2\n";

    // Random Number Generator
    mt19937_64 rng(42);
    uniform_real_distribution<double> dist(0.0, 10000.0);

    // Generate Data
    for (long long i = 0; i < size; i++)
    {
        double value1 = dist(rng);
        double value2 = dist(rng);

        file << value1 << "," << value2 << "\n";
    }

    file.close();

    cout << filename << " generated successfully!" << endl;
    cout << "Total Rows: " << size << endl;
    cout << "----------------------------------" << endl;
}


int main()
{
    cout << "========== DATASET GENERATOR ==========" << endl;
    cout << endl;

    
    generateDataset("dataset/small.csv", 1000000);
    generateDataset("dataset/medium.csv", 10000000);
    generateDataset("dataset/large.csv", 100000000);

    cout << endl;
    cout << "All datasets generated successfully!" << endl;

    return 0;
}