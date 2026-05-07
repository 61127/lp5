#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <ctime>

using namespace std;

// Function to find minimum
void findMin(int arr[], int n)
{
    int min_val = arr[0];

#pragma omp parallel for reduction(min : min_val)
    for (int i = 0; i < n; i++)
    {
        if (arr[i] < min_val)
            min_val = arr[i];
    }

    cout << "\nMinimum = " << min_val;
}

// Function to find maximum
void findMax(int arr[], int n)
{
    int max_val = arr[0];

#pragma omp parallel for reduction(max : max_val)
    for (int i = 0; i < n; i++)
    {
        if (arr[i] > max_val)
            max_val = arr[i];
    }

    cout << "\nMaximum = " << max_val;
}

// Function to find sum and average
void findSumAvg(int arr[], int n)
{
    int sum = 0;

#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < n; i++)
    {
        sum += arr[i];
    }

    double avg = (double)sum / n;

    cout << "\nSum = " << sum;
    cout << "\nAverage = " << avg;
}

int main()
{

    omp_set_num_threads(4);

    int n;
    cout << "Enter number of elements: ";
    cin >> n;

    int arr[n];

    // Generate random values
    srand(time(0));
    for (int i = 0; i < n; i++)
    {
        arr[i] = rand() % 100;
    }

    cout << "\nArray: ";
    for (int i = 0; i < n; i++)
    {
        cout << arr[i] << " ";
    }

#pragma omp parallel
    {
#pragma omp single
        cout << "Threads = " << omp_get_num_threads() << endl;
    }

    double start = omp_get_wtime();

    findMin(arr, n);
    findMax(arr, n);
    findSumAvg(arr, n);

    double end = omp_get_wtime();

    cout << "\nExecution Time = " << (end - start) << " seconds\n";

    return 0;
}