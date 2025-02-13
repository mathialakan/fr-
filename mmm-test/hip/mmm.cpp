#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include "../../fraction/fr.h"
#include <unordered_map>

#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>

#if defined(X86)
#include <immintrin.h>
#endif
#if defined(OMP)
#include <omp>
#elif defined(OMP_OL)
#define team_size 256
#endif
using namespace std;


#define EMAX 10
#define WMAX 10
#define NMAX 10
#define DMAX 10

/* Using c-style pointer 1D arrays*/
template<typename et>
bool fill_mat(et* A, int M, int N){
    srand(time(0));

    for(int i=0; i <M; ++i)
    for(int j=0; j <N; ++j)
        A[N*i +j] = 1 + (rand()% EMAX);

    return true;
}

//template<typename et>
bool fill_mat_fr(fraction* A, int M, int N){
    srand(time(0));

    for(int i=0; i <M; ++i){
    for(int j=0; j <N; ++j){
        fraction e(1 + (rand()% NMAX), 1 + (rand()% DMAX));
        A[N*i +j] = e;
    }
    }
    return true;
}

bool fill_mat_mfr(mixed* A, int M, int N){
    srand(time(0));

    for(int i=0; i <M; ++i){
    for(int j=0; j <N; ++j){
        mixed e( 1 + (rand()% WMAX), fraction(1 + (rand()% NMAX), 1 + (rand()% DMAX)));
        A[N*i +j] = e;
    }
    }
    return true;
}

template<typename et>
bool print_mat(et* A, int M, int N){
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++)
            cout<< "\t" <<A[N*i+j];
        cout <<endl;
    }
    return true;
}

/* Using c-style pointer 2D arrays */

template<typename et>
bool fill_mat(et** A){

    int M = sizeof(A)/sizeof(A[0]);
    int N = sizeof(A[0])/sizeof(A[0][0]);

    for(int i=0; i <M; ++i)
    for(int j=0; j <N; ++j)
        A[i][j] = 1+ (rand()%EMAX);
    
    return true;
}

/* Using vector */
template<typename et>
int fill_mat(vector<vector<et>> &A){

    srand(time(0));  //Seeding

    int rows =  A.size();
    if (rows==0) return 1;
    int cols = A[0].size();

    for(int i=0; i< rows; ++i)
    for(int j=0; j< cols; ++j)
        A[i][j] = 1 + (rand()%EMAX);

    return 0;
}
template<typename et>
int read_mat( vector<vector<et> > &A){

    int rows = A.size();
    if (rows==0) return 1;
    int cols = A[0].size();

    for(int i=0; i< rows; ++i)
    for(int j=0; j< cols; ++j)
        cin >> A[i][j] ;

    return 0;
}

template<typename et>
int print_mat(vector<vector<et> > A){

    int rows = A.size();
    if (rows==0) return 1;

    int cols = A[0].size();
    for(int i=0; i< rows; ++i){
        for(int j=0; j< cols; ++j)
            cout<< "\t" << A[i][j] ;
        cout<< endl;
    }
    return 0;
}

template<typename et>
__global__ void mat_mul(et* A, et* B, et* C, int M, int K, int N){
    int rows = blockIdx.y*blockDim.y + threadIdx.y;
    int cols = blockIdx.x*blockDim.x + threadIdx.x;

        et s;
    if ( rows < M && cols < N){
        s = 0;
        for (int k=0; k < K; ++k)
        s += A[rows*K +k] *B[k*N +cols];
    }
    C[rows*N +cols] = s;
}

template<typename et>
__global__ void mat_mul_(et* A, et* B, et* C, int M, int K, int N){
    int rows = blockIdx.y*blockDim.y + threadIdx.y;
    int cols = blockIdx.x*blockDim.x + threadIdx.x;

    for (int i = rows; i < M; i += blockDim.y*gridDim.y)
    for (int j = cols; j < N; j += blockDim.x*gridDim.x){
        et s;
        s = 0;
	//printf("s : %d \n", s);
        for (int k=0; k < K; ++k){
            //s = B[j*N +k];
            //s = A[i*K +k] ;
            s += A[i*K +k] *B[j*N +k];
	//printf("s : %d \n", s);
	}
        C[i*N +j] = s;
    }
}

template<typename et>
void mat_mul_hip(size_t grid_sz, size_t block_sz, et* A, et* B, et* C, int M, int K, int N){

        //mat_mul<et><<< grid_sz, block_sz>>>(A, B, C, M, K, N);
        mat_mul_<et><<< grid_sz, block_sz>>>(A, B, C, M, K, N);
}

template<typename et>
unordered_map<string, double> test_mm( int m, int k, int n){

    unordered_map<string, double> time_local;
    int nele = m*n;
    size_t block_sz = (nele < 1024) ? nele : 1024;
    size_t grid_sz = (nele +1)/block_sz;
    et* dA; et* dB; et* dC;
    et* A; et* B; et* C;
    hipMallocHost( (void**)&A, m*k* sizeof(et));
    hipMallocHost( (void**)&B, k*n* sizeof(et));
    hipMallocHost( (void**)&C, m*n* sizeof(et));

    hipMalloc( (void**)&dA, m*k* sizeof(et));
    hipMalloc( (void**)&dB, k*n* sizeof(et));
    hipMalloc( (void**)&dC, m*n* sizeof(et));

    auto start_time = chrono::steady_clock::now();
    fill_mat(A, m, k);
    fill_mat(B, k, n);
    auto end_time = chrono::steady_clock::now();
    time_local["fill_mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    print_mat<et>(A, m, k);
    cout<<endl;
    print_mat<et>(B, k, n);
    end_time = chrono::steady_clock::now();
    time_local["print_pre-mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    hipMemcpy(dA, A, m*k* sizeof(et), hipMemcpyHostToDevice);
    hipMemcpy(dB, B, k*n* sizeof(et), hipMemcpyHostToDevice);

    end_time = chrono::steady_clock::now();
    time_local["h2d"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();

    mat_mul_hip<et>(grid_sz, block_sz, dA, dB, dC, m, k, n);
    end_time = chrono::steady_clock::now();
    time_local["comp_mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();

    hipMemcpy( C, dC, m*n* sizeof(et), hipMemcpyDeviceToHost);
    end_time = chrono::steady_clock::now();
    time_local["d2h"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    cout << "Result" << endl;
    print_mat<et>(C, m, n);
    end_time = chrono::steady_clock::now();
    time_local["print_post-mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();

    //Cleaning up
    hipFreeHost(A);
    hipFreeHost(B);
    hipFreeHost(C);
    hipFree(dA);
    hipFree(dB);
    hipFree(dC);

    return time_local;

}

template<typename et>
unordered_map<string, double> test_mm_fr( int m, int k, int n){

    unordered_map<string, double> time_local;
    int nele = n*m;
    size_t block_sz = (nele < 1024) ? nele : 1024;
    size_t grid_sz = (nele +1)/block_sz;
    et* dA; et* dB; et* dC;
    et* A; et* B; et* C;
    hipMallocHost( (void**)&A, m*k* sizeof(et));
    hipMallocHost( (void**)&B, k*n* sizeof(et));
    hipMallocHost( (void**)&C, m*n* sizeof(et));

    hipMalloc( (void**)&dA, m*k* sizeof(et));
    hipMalloc( (void**)&dB, k*n* sizeof(et));
    hipMalloc( (void**)&dC, m*n* sizeof(et));

    auto start_time = chrono::steady_clock::now();
    fill_mat_fr(A, m, k);
    fill_mat_fr(B, k, n);
    auto end_time = chrono::steady_clock::now();
    time_local["fill_mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    print_mat<et>(A, m, k);
    cout<<endl;
    print_mat<et>(B, k, n);
    end_time = chrono::steady_clock::now();
    time_local["print_pre-mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    hipMemcpy(dA, A, m*k* sizeof(et), hipMemcpyHostToDevice);
    hipMemcpy(dB, B, k*n* sizeof(et), hipMemcpyHostToDevice);

    end_time = chrono::steady_clock::now();
    time_local["h2d"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();

    mat_mul_hip<et>(grid_sz, block_sz, dA, dB, dC, m, k, n);
    end_time = chrono::steady_clock::now();
    time_local["comp_mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();

    hipMemcpy( C, dC, m*n* sizeof(et), hipMemcpyDeviceToHost);
    end_time = chrono::steady_clock::now();
    time_local["d2h"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    cout << "Result" << endl;
    print_mat<et>(C, m, n);
    end_time = chrono::steady_clock::now();
    time_local["print_post-mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();

    //Cleaning up
    hipFreeHost(A);
    hipFreeHost(B);
    hipFreeHost(C);
    hipFree(dA);
    hipFree(dB);
    hipFree(dC);

    return time_local;

}

template<typename et>
unordered_map<string, double> test_mm_mfr( int m, int k, int n){

    unordered_map<string, double> time_local;
    int nele = m*n;
    size_t block_sz = (nele < 1024) ? nele : 1024;
    size_t grid_sz = (nele +1)/block_sz;
    et* dA; et* dB; et* dC;
    et* A; et* B; et* C;
    hipMallocHost( (void**)&A, m*k* sizeof(et));
    hipMallocHost( (void**)&B, k*n* sizeof(et));
    hipMallocHost( (void**)&C, m*n* sizeof(et));

    hipMalloc( (void**)&dA, m*k* sizeof(et));
    hipMalloc( (void**)&dB, k*n* sizeof(et));
    hipMalloc( (void**)&dC, m*n* sizeof(et));

    auto start_time = chrono::steady_clock::now();
    fill_mat_mfr(A, m, k);
    fill_mat_mfr(B, k, n);
    auto end_time = chrono::steady_clock::now();
    time_local["fill_mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    print_mat<et>(A, m, k);
    cout<<endl;
    print_mat<et>(B, k, n);
    end_time = chrono::steady_clock::now();
    time_local["print_pre-mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    hipMemcpy(dA, A, m*k* sizeof(et), hipMemcpyHostToDevice);
    hipMemcpy(dB, B, k*n* sizeof(et), hipMemcpyHostToDevice);

    end_time = chrono::steady_clock::now();
    time_local["h2d"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();

    mat_mul_hip<et>(grid_sz, block_sz, dA, dB, dC, m, k, n);
    end_time = chrono::steady_clock::now();
    time_local["comp_mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();

    hipMemcpy( C, dC, m*n* sizeof(et), hipMemcpyDeviceToHost);
    end_time = chrono::steady_clock::now();
    time_local["d2h"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    cout << "Result" << endl;
    print_mat<et>(C, m, n);
    end_time = chrono::steady_clock::now();
    time_local["print_post-mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();

    //Cleaning up
    hipFreeHost(A);
    hipFreeHost(B);
    hipFreeHost(C);
    hipFree(dA);
    hipFree(dB);
    hipFree(dC);

    return time_local;

}

void efficiency_analysis(unordered_map<string, double> time_map ){

    cout<< "----------------------------------" << endl;
    cout<< "Task" << "\t" << "Time (micro-sec)" << endl;
    cout<< "----------------------------------" << endl;
    for(auto& pair : time_map)
        cout<< pair.first << "\t" << pair.second << endl;
    cout<< "----------------------------------" << endl <<endl;

}

void efficiency_analysis(unordered_map<string, double> time_map , unordered_map<string, int> ndaccesses, size_t tsize, int ops){

    cout<< " Type size (in Bytes) : " << tsize  << endl;
    cout<< "----------------------------------" << endl;
    cout<< "Task" << "\t" << "Time (micro-sec)" << endl;
    cout<< "----------------------------------" << endl;
    for(auto& pair : time_map)
        cout<< pair.first << "\t" << pair.second << endl;
    cout<< "----------------------------------" << endl <<endl;

    cout<< "----------------------------------" << endl;
    cout<< "Task" << "\t" << "Bandwidth [GB/s]" << "\t" << "Arithmetic Intensity [OPS/Bytes]" << endl;
    cout<< "----------------------------------" << endl;

    for(auto& pair : time_map){
        auto seconds = pair.second * 1.e-6;
        auto nbytes_accesses = ndaccesses[pair.first]*tsize;
         // Bandwidth [GB/s] = Data accessed in GB / Consumed time in sec
        auto bwidth = (seconds>0) ? (double)(nbytes_accesses/seconds)* 1.e-9 : -1; // GB/s
         // Arithmetic Intensity (ops/bytes - FLOPS/B) = number of FLOPS / number of byte accesses
        auto ai = (double)(ops/nbytes_accesses);
        cout<< pair.first << "\t" << bwidth << "\t" << ai << endl;
    }       
    cout<< "----------------------------------" << endl;

    
}

void test_all(int n, int k, int m){

    unordered_map<string, double> time_map;
    auto start_time = chrono::steady_clock::now();
    auto time_int = test_mm<int>(m, k, n); cout<< "Finished MMM using 1-D array of integers" << endl; //Test 1-D array
    auto end_time = chrono::steady_clock::now();
    time_map["1d-int"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;
    auto time_fl = test_mm<float>(m, k, n); cout<< "Finished MMM using 1-D array of floats" << endl; //Test 1-D array
    end_time = chrono::steady_clock::now();
    time_map["1d-float"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;
    auto time_fr = test_mm_fr<fraction>(m, k, n); cout<< "Finished MMM using 1-D array of fractions" << endl; //Test Fraction
    end_time = chrono::steady_clock::now();
    time_map["1d-fraction"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;
    auto time_mfr = test_mm_mfr<mixed>(m, k, n); cout<< "Finished MMM using 1-D array of mixed fractions" << endl; //Test Fraction
    end_time = chrono::steady_clock::now();
    time_map["1d-mixed"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    unordered_map<string, int> ndaccesses;
    ndaccesses["fill_mat"] = m*k +k*n;
    ndaccesses["print_pre-mat"] = m*k +k*n;
    ndaccesses["comp_mat"] = m*k +k*n +2*n*m;  // read data from A(m,k), B(k,n), and C(m,n) and write it to C(m,n)
    ndaccesses["print_post-mat"] = n*m;

    size_t niterations = m*k*n;  // traditional matrix multiplication algo O(N^3)
    unordered_map<string, int> ops;
    ops["1d-int"] = 2*niterations;
    ops["1d-float"] = 2*niterations;
    ops["1d-fraction"] = 12*niterations;  //( 7 ops for addition and 5 ops for multiplication)
    ops["1d-mixed"] = 43*niterations;  //(23 ops for addition and 20 ops for multiplication)

    cout<< endl << "Total Time " << endl;
    efficiency_analysis(time_map);
    cout<< endl << "1D-int Time " << endl;
    efficiency_analysis(time_int, ndaccesses, sizeof(int), ops["1d-int"]);
    cout<< endl << "1D-float Time " << endl;
    efficiency_analysis(time_fl, ndaccesses, sizeof(float), ops["1d-float"]);
    cout<< endl << "1D-fraction Time " << endl;
    efficiency_analysis(time_fr, ndaccesses, sizeof(fraction), ops["1d-fraction"]);
    cout<< endl << "1D-mixed Time " << endl;
    efficiency_analysis(time_mfr, ndaccesses, sizeof(mixed), ops["1d-mixed"]);

}

int main(int argc, char* argv[]){
    
    int m, k, n;
    switch (argc){
            case 1:
                 m = 100; n = 100; k=100; break;
            case 2:
                 m = atoi(argv[1]); k = m; n = m; break;
            case 3:
                 m = atoi(argv[1]); k = atoi(argv[2]); n = m; break;
            case 4:
                 m = atoi(argv[1]); k = atoi(argv[2]); n = atoi(argv[3]); break;
            case 5:
                if(atoi(argv[2]) != atoi(argv[3])) {
		       	cout<< "Number of columns of the first matrix should be equal to the number of rows of the second matrix" << endl;
                  	return 0;
		}else
		{
			m = atoi(argv[1]); k = atoi(argv[2]); n = atoi(argv[4]); break;
		}	
            default: break;
    }           

    // 2D-Vector of floats
    /*
    vector<vector<float> > A(m, vector<float>(k));
    vector<vector<float> > B(k, vector<float>(n));

    //if (read_mat(A) == 1) cout<< "Reading is failed" << endl;
    //if (read_mat(B) == 1) cout<< "Reading is failed" << endl;
    fill_mat(A);
    fill_mat(B);

    auto C = mat_mul(A,B);
    cout << "Result" << endl;
    print_mat((vector<vector<float> >)C);
    */
//*
      string test_case;
    int ops_it;
    unordered_map<string, double> time_map;
    auto start_time = chrono::steady_clock::now();
    size_t sizet;
    unordered_map<string, double> time_exe;
// default
#if defined(INT_1D)
    test_case = "1-D array of integers";
    ops_it = 2;
    sizet = sizeof(int);
    time_exe = test_mm<int>(m, k, n);
#elif defined(FLOAT_1D)
    test_case = "1-D array of floats";
    ops_it = 2;
    sizet = sizeof(float);
    time_exe = test_mm<float>(m, k, n);
#elif defined(FRACTION_1D)
    test_case = "1-D array of fractions";
    ops_it = 12;
    sizet = sizeof(fraction);
    time_exe = test_mm_fr<fraction>(m, k, n);
#elif defined(MIXED_1D)
    test_case = "1-D array of mixed";
    ops_it = 43;
    sizet = sizeof(int);
    time_exe = test_mm_mfr<mixed>(m, k, n);
#elif defined(DOUBLE_1D) 
    test_case = "1-D array of double";
    ops_it = 2;
    sizet = sizeof(int);
    time_exe = test_mm_fr<double>(m, k, n);
#endif  
    auto end_time = chrono::steady_clock::now();
    auto time_total = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    cout<< "Finished MMM using " << test_case << endl; //Test 1-D array

    unordered_map<string, int> ndaccesses;
    ndaccesses["fill_mat"] = m*k +k*n;
    ndaccesses["h2d"] = m*k +k*n;
    ndaccesses["print_pre-mat"] = n*m;
    ndaccesses["comp_mat"] = m*k +k*n +2*n*m;  // read data from A(m,k), B(k,n), and C(m,n) and write it to C(m,n)
    ndaccesses["print_post-mat"] = n*m;
    ndaccesses["d2h"] = n*m;
        
    size_t niterations = m*k*n;  // traditional matrix multiplication algo O(N^3)
    int ops = ops_it*niterations;

    cout<< endl << " Time " << endl;
    efficiency_analysis(time_exe, ndaccesses, sizet, ops); 
    //-------------- HIP  ----------------//
    //------------------------------------//
    return 0;
}
