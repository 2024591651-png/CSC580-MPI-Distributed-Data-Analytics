
// ==========================================================
// CSC580 MPI Project - Sequential Analytics
// Includes:
// Read CSV, Mean, Min, Max, Variance, Std Dev,
// Histogram, Sorting, Pearson Correlation,
// Moving Average, Outlier Detection,
// Execution Timing, CSV Output
// ==========================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <chrono>

using namespace std;

vector<double> value1, value2;

// ---------- Statistics ----------
double calculateMean(const vector<double>& d){
    double s=0; for(double v:d) s+=v; return s/d.size();
}
double calculateMinimum(const vector<double>& d){
    return *min_element(d.begin(),d.end());
}
double calculateMaximum(const vector<double>& d){
    return *max_element(d.begin(),d.end());
}
double calculateVariance(const vector<double>& d){
    double m=calculateMean(d),s=0;
    for(double v:d){ double x=v-m; s+=x*x; }
    return s/d.size();
}
double calculateStandardDeviation(const vector<double>& d){
    return sqrt(calculateVariance(d));
}
double calculatePearsonCorrelation(const vector<double>& x,const vector<double>& y){
    double mx=calculateMean(x), my=calculateMean(y);
    double num=0,dx2=0,dy2=0;
    for(size_t i=0;i<x.size();i++){
        double dx=x[i]-mx, dy=y[i]-my;
        num+=dx*dy;
        dx2+=dx*dx;
        dy2+=dy*dy;
    }
    double den=sqrt(dx2*dy2);
    return den==0?0:num/den;
}

// ---------- Histogram ----------
void calculateHistogram(const vector<double>& d){
    vector<int> hist(10,0);
    for(double v:d){
        int b=(int)(v/1000);
        if(b>9)b=9;
        hist[b]++;
    }
    cout<<"\n========== HISTOGRAM ==========\n";
    for(int i=0;i<10;i++)
        cout<<setw(4)<<i*1000<<" - "<<setw(5)<<(i+1)*1000-1<<" : "<<hist[i]<<'\n';
}

// ---------- Sorting ----------
void sortData(const vector<double>& d){
    vector<double> s=d;
    sort(s.begin(),s.end());
    cout<<"\n========== SORTED DATA ==========\n";
    cout<<"Smallest 10 values:\n";
    for(size_t i=0;i<10 && i<s.size();i++) cout<<s[i]<<'\n';
    cout<<"\nLargest 10 values:\n";
    size_t start=s.size()>10?s.size()-10:0;
    for(size_t i=start;i<s.size();i++) cout<<s[i]<<'\n';
}

// ---------- Moving Average ----------
vector<double> calculateMovingAverage(const vector<double>& d,int w){
    vector<double> out;
    if(d.size()<static_cast<size_t>(w)) return out;
    for(size_t i=0;i<=d.size()-w;i++){
        double sum=0;
        for(int j=0;j<w;j++) sum+=d[i+j];
        out.push_back(sum/w);
    }
    return out;
}
void displayMovingAverage(const vector<double>& d){
    auto m=calculateMovingAverage(d,3);
    cout<<"\n========== MOVING AVERAGE ==========\n";
    for(size_t i=0;i<10 && i<m.size();i++) cout<<m[i]<<'\n';
}

// ---------- Outliers ----------
void detectOutliers(const vector<double>& d){
    double mean=calculateMean(d), sd=calculateStandardDeviation(d);
    cout<<"\n========== OUTLIERS (|Z|>3) ==========\n";
    int c=0;
    if(sd==0){ cout<<"None\n"; return; }
    for(size_t i=0;i<d.size();i++){
        double z=(d[i]-mean)/sd;
        if(fabs(z)>3){
            cout<<"Index "<<i<<" Value "<<d[i]<<" Z="<<z<<'\n';
            c++;
        }
    }
    if(c==0) cout<<"No outliers detected.\n";
}

void exportCSV(){
    ofstream out("results.csv");
    out<<"Metric,Value\n";
    out<<"Mean Value1,"<<calculateMean(value1)<<"\n";
    out<<"Mean Value2,"<<calculateMean(value2)<<"\n";
    out<<"Minimum Value1,"<<calculateMinimum(value1)<<"\n";
    out<<"Maximum Value1,"<<calculateMaximum(value1)<<"\n";
    out<<"Variance Value1,"<<calculateVariance(value1)<<"\n";
    out<<"StdDev Value1,"<<calculateStandardDeviation(value1)<<"\n";
    out<<"Pearson Correlation,"<<calculatePearsonCorrelation(value1,value2)<<"\n";
}
void readCSV(string filename, vector<double>& value1, vector<double>& value2)
{
    ifstream file(filename);

    if(!file)
    {
        cout << "Cannot open " << filename << endl;
        exit(1);
    }

    string line, a, b;

    getline(file, line); // Skip header

    while(getline(file, line))
    {
        stringstream ss(line);

        getline(ss, a, ',');
        getline(ss, b);

        value1.push_back(stod(a));
        value2.push_back(stod(b));
    }

    file.close();
}
int main(int argc, char* argv[]){
    string filename;

    if(argc < 2)
    {
        cout << "Usage: sequential_analytics <dataset_size>" << endl;
        return 1;
    }

    int datasetSize = stoi(argv[1]);

    if(datasetSize == 1000000)
        filename = "dataset/small.csv";
    else if(datasetSize == 10000000)
        filename = "dataset/medium.csv";
    else if(datasetSize == 100000000)
        filename = "dataset/large.csv";
    else
    {
        cout << "Invalid dataset size!" << endl;
        return 1;
    }

    auto start = chrono::high_resolution_clock::now();

    readCSV(filename, value1, value2);

    cout<<"Records: "<<value1.size()<<"\n";
    cout<<"Mean1: "<<calculateMean(value1)<<'\n';
    cout<<"Mean2: "<<calculateMean(value2)<<'\n';
    cout<<"Min1: "<<calculateMinimum(value1)<<'\n';
    cout<<"Max1: "<<calculateMaximum(value1)<<'\n';
    cout<<"Variance1: "<<calculateVariance(value1)<<'\n';
    cout<<"StdDev1: "<<calculateStandardDeviation(value1)<<'\n';
    cout<<"Pearson Correlation: "<<calculatePearsonCorrelation(value1,value2)<<'\n';

    calculateHistogram(value1);
    sortData(value1);
    displayMovingAverage(value1);
    detectOutliers(value1);

    exportCSV();

    auto end=chrono::high_resolution_clock::now();
    auto ms=chrono::duration_cast<chrono::milliseconds>(end-start);

    cout<<"\nExecution Time: "<<ms.count()<<" ms\n";
    cout<<"Results saved to results.csv\n";
    return 0;
}
