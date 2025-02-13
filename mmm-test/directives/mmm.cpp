#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include "../../fraction/fr.h"
#include <unordered_map>

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
et* mat_mul(et* A, et* B, int M, int K, int N){

    size_t nel = N*M;
    et* C = (et*)malloc(nel*sizeof(et));
    #if defined(OMP_OL)
        #pragma omp target enter data map(alloc: C[:nel])
    #elif defined(OACC)
        #pragma acc enter data create(C[:nel])
    #endif

    #if defined(OMP)
        #pragma omp parallel for collapse(2) schedule(dynamic)
    #elif defined(OMP_OL)
        #pragma omp target teams distribute parallel for collapse(2) \
            thread_limit(team_size) num_teams((N*M +1)/team_size) 
    #elif defined(OACC)
        #pragma acc parallel loop present(A, B, C) gang worker \
                num_workers(team_size) vector_length(32)
    #endif
    for(int i=0; i<M; i++)
        for(int j=0; j<N; j++){
            C[N*i +j]= 0;
            for(int k=0; k<K; k++)
                C[N*i +j] += A[K*i + k] *B[N*k +j];  // 2 ops for prime type, 12 ops (7+5) for fraction, 47 ops (23 +24) for mixed numbers
        }
    return C;
}

template<typename et>
et* mat_mul_i0(et* A, et* B, int M, int K, int N){

    size_t nel = N*M;
    et* C = (et*)malloc(nel*sizeof(et));
    C = {0};
    #if defined(OMP_OL)
        #pragma omp target enter data map(alloc: C[:nel])
    #elif defined(OACC)
        #pragma acc enter data create(C[:nel])
    #endif

    #if defined(OMP)
        #pragma omp parallel for collapse(2) schedule(dynamic)
    #elif defined(OMP_OL)
        #pragma omp target teams distribute parallel for collapse(2) \
            thread_limit(team_size) num_teams((N*M +1)/team_size) 
    #elif defined(OACC)
        #pragma acc parallel loop present(A, B, C) gang worker \
                num_workers(team_size) vector_length(32)
    #endif
    for(int i=0; i<M; i++)
        for(int j=0; j<N; j++){
            //C[M*i +j]= 0;
            #if defined(OMP)
            #pragma omp simd
            #elif defined(OMP_OL)
            #pragma omp simd
            #elif defined(OACC)
            #pragma acc loop vector
            #endif
            for(int k=0; k<K; k++)
                C[N*i +j] += A[K*i + k] *B[N*k +j]; 
        }
    return C;
}


template<typename et>
et* mat_mul_avx(et* A, et* B, int M, int K, int N, int simd_len){  //simd_len = 8
  
    size_t nel = N*M;
    et* C = (et*)malloc(nel*sizeof(et));
    C = {0};
    #if defined(OMP_OL)
        #pragma omp target enter data map(alloc: C[:nel])
    #elif defined(OACC)
        #pragma acc enter data create(C[:nel])
    #endif

    #if defined(OMP)
        #pragma omp parallel for collapse(2) schedule(dynamic)
    #elif defined(OMP_OL)
        #pragma omp target teams distribute parallel for collapse(3) \
            thread_limit(team_size) num_teams((N*M +1)/team_size) 
    #elif defined(OACC)
        #pragma acc parallel loop present(A, B, C) gang worker \
                num_workers(team_size) vector_length(32)
    #endif
    for(int i=0; i<M; i++)
        for(int j=0; j<N; j++){
            for(int k=0; k<K; k+=simd_len) 
            //Vectorizing using AVX intrinsics
            #if defined(X86)
                __m256 av = _mm256_loadu_ps(A+ (K*i + k)); 
                __m256 bv = _mm256_loadu_ps(B+ (M*k +j));
                __m256 cv = _mm256_add_ps(av,bv);
                _mm256_storeu_ps(C +(M*i +j), cv); 
            #endif
            C[N*i +j] += A[K*i + k] *B[N*k +j]; 
        }
    return C;
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
template<typename et>
et** mat_maul(et** A, et** B){

    //int size_bytes = sizeof(A);
    int M = sizeof(A)/sizeof(A[0]);
    int K = sizeof(A[0])/sizeof(A[0][0]);
    int N = sizeof(B)/sizeof(B[0])/sizeof(B[0][0]);

    et** C = (et**) malloc(M*sizeof(et*));
    for(int i=0; i<M; i++)
        C[i] = (et*)malloc(N*sizeof(et));

    #if defined(OMP_OL)
        #pragma omp target enter data map(alloc: C[:M][:N])
    #elif defined(OACC)
        #pragma acc enter data create(C[:M][:N])
    #endif

    #if defined(OMP)
        #pragma omp parallel for collapse(2) schedule(dynamic) 
    #elif defined(OMP_OL)
        #pragma omp target teams distribute parallel for \
                thread_limit(team_size) num_teams((N*M +1)/team_size)
    #elif defined(OACC)
        #pragma acc parallel loop
    #endif
    for(int i=0; i<M; i++)
        for(int j=0; j<N; j++){
            C[i][j] = 0;
            for(int k=0; k<K; k++)
                C[i][j] += A[i][k]*B[k][j];
        }
    return C;        

}

/* Using cpp-style 1D pointer arrays */
template<typename et>
et* mat_mul_cpp(et* A, et* B, int M, int K, int N){

    size_t nel = N*M;
    et* C = new et[nel];
    
    #if defined(OMP_OL)
        #pragma omp target enter data map(alloc: C[:nel])
    #elif defined(OACC)
        #pragma acc enter data create(C[:nel])
    #endif

    #if defined(OMP)
        #pragma omp parallel for collapse(2) schedule(dynamic)
    #elif defined(OMP_OL)
        #pragma omp target teams distribute parallel for \
                thread_limit(team_size) num_teams((N*M +1)/team_size)
    #elif defined(OACC)
        #pragma acc parallel loop
    #endif
    for(int i=0; i<M; i++)
    for(int j=0; j<N; j++){
        C[M*i +j] = 0;
        for(int k=0; k<K; k++)
            C[N*i +j] += A[K*i+k]*B[N*k+j];
    }
    return C;

}

/* Using cpp-style 2D pointer arrays */
template<typename et>
et** mat_maul_cpp(et** A, et** B){

    int M = sizeof(A)/sizeof(A[0]);
    int K = sizeof(A[0])/sizeof(A[0][0]);
    int N = sizeof(B)/sizeof(B[0])/sizeof(B[0][0]);

    et** C =  new et*[M];
    for (int i=0; i<M; i++)
        C[i] = new et[N];

    #if defined(OMP_OL)
        #pragma omp target enter data map(alloc: C[:M][:N])
    #elif defined(OACC)
        #pragma acc enter data create(C[:M][:N])
    #endif

    #if defined(OMP)
        #pragma omp parallel for collapse(2) schedule(dynamic)
    #elif defined(OMP_OL)
        #pragma omp target teams distribute parallel for \
                thread_limit(team_size) num_teams((N*M +1)/team_size)
    #elif defined(OACC)
        #pragma acc parallel loop
    #endif
    for (int i=0; i<M; i++)
        for( int j=0; j<N; j++){
            C[i][j]=0;
            for(int k=0; k<K; k++)
                C[i][j] += A[i][k] * B[k][j]; 
        }
        
    return C;
}

/* Using 1D vector*/
template<typename et>
vector<et> mat_mul(vector<et> A, vector<et> B){

    
}


/* Using 2D vector*/
template<typename et>
vector<vector<et> > mat_mul(vector<vector<et> > A, vector<vector<et> > B){

    //if (A.size()==0 || B.size()==0) return 0;
    int M = A.size();
    int K = A[0].size();
    int N = B[0].size();
    //if (K != B.size()) return 0;
    vector<vector<et> > C(M, vector<et>(N));
    #if defined(OMP_OL)
        #pragma omp target enter data map(alloc: C)
    #elif defined(OACC)
        #pragma acc enter data create(C)
    #endif

    #if defined(OMP)
        #pragma omp parallel for collapse(2) schedule(dynamic)
    #elif defined(OMP_OL)
        #pragma omp target teams distribute parallel for \
                thread_limit(team_size) num_teams((N*M +1)/team_size)
    #elif defined(OACC)
        #pragma acc parallel loop
    #endif  
    for(int i=0; i< M; ++i)
        for(int j=0; j< N; ++j){
           C[i][j] = 0;
           for(int k=0; k< K; ++k)
               C[i][j] += A[i][k] * B[k][j]; 
        }
    return C;
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
unordered_map<string, double> test_mm(int m, int k, int n){

    unordered_map<string, double> time_local;

    et* A = (et*)malloc( m*k* sizeof(et));
    et* B = (et*)malloc( k*n* sizeof(et));
    auto start_time = chrono::steady_clock::now();

    fill_mat(A, m, k);
    fill_mat(B, k, n);
    #if defined(OMP_OL)
        #pragma omp target enter data map(to:A[0:m*k],B[0:k*n])
    #elif defined(OACC)
        #pragma acc enter data copyin(A[:m*k],B[:k*n])
    #endif
    auto end_time = chrono::steady_clock::now();
    time_local["fill_mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    print_mat<et>(A, m, k);
    cout<<endl;
    print_mat<et>(B, k, n);
    end_time = chrono::steady_clock::now();
    time_local["print_pre-mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    auto C = mat_mul<et>(A, B, m, k, n);
    end_time = chrono::steady_clock::now();
    time_local["comp_mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    cout << "Result" << endl;
    #if defined(OMP_OL)
        #pragma omp target update from(C[0:n*m])
    #elif defined(OACC)
        #pragma acc update self(C[0:n*m])
    #endif
    print_mat<et>(C, m, n);
    end_time = chrono::steady_clock::now();
    time_local["print_post-mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    //start_time = end_time;

    #if defined(OMP_OL)
        #pragma omp target exit data map(delete:A[0:m*k],B[0:k*n],C[0:m*n])
    #elif defined(OACC)
        #pragma acc exit data delete(A[:m*k],B[:k*n],C[0:m*n])
    #endif

    return time_local;
}

template<typename et>
unordered_map<string, double> test_mm_fr(int m, int k, int n){

    unordered_map<string, double> time_local;
    //et* A; 
    //et* B;

    et* A = (et*)malloc( m*k* sizeof(et));
    et* B = (et*)malloc( k*n* sizeof(et));
    auto start_time = chrono::steady_clock::now();
    fill_mat_fr(A, m, k);
    fill_mat_fr(B, k, n); 

    #if defined(OMP_OL)
        #pragma omp target enter data map(to:A[0:m*k],B[0:k*n])
    #elif defined(OACC)
        #pragma acc enter data copyin(A[:m*k],B[:k*n])
    #endif

    auto end_time = chrono::steady_clock::now();
    time_local["fill_mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    print_mat<et>(A, m, k);
    cout<<endl;
    print_mat<et>(B, k, n);
    end_time = chrono::steady_clock::now();
    time_local["print_pre-mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    auto C = mat_mul<et>(A, B, m, k, n);
    end_time = chrono::steady_clock::now();
    time_local["comp_mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    cout << "Result" << endl;
    #if defined(OMP_OL)
        #pragma omp target update from(C[0:n*m])
    #elif defined(OACC)
        #pragma acc update self(C[0:n*m])
    #endif
    print_mat<et>(C, m, n);
    end_time = chrono::steady_clock::now();
    time_local["print_post-mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    //start_time = end_time;

    #if defined(OMP_OL)
        #pragma omp target exit data map(delete:A[0:m*k],B[0:k*n],C[0:m*n])
    #elif defined(OACC)
        #pragma acc exit data delete(A[:m*k],B[:k*n],C[0:m*n])
    #endif
    return time_local;
}

template<typename et>
unordered_map<string, double> test_mm_mfr(int m, int k, int n){

    unordered_map<string, double> time_local;

    et* A = (et*)malloc( m*k* sizeof(et));
    et* B = (et*)malloc( k*n* sizeof(et));
    auto start_time = chrono::steady_clock::now();
    fill_mat_mfr(A, m, k);
    fill_mat_mfr(B, k, n); 
    #if defined(OMP_OL)
        #pragma omp target enter data map(to:A[0:m*k],B[0:k*n])
    #elif defined(OACC)
        #pragma acc enter data copyin(A[:m*k],B[:k*n])
    #endif

    auto end_time = chrono::steady_clock::now();
    time_local["fill_mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;
    
    print_mat<et>(A, m, k);
    cout<<endl;
    print_mat<et>(B, k, n);
    end_time = chrono::steady_clock::now();
    time_local["print_pre-mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    auto C = mat_mul<et>(A, B, m, k, n);
    end_time = chrono::steady_clock::now();
    time_local["comp_mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    cout << "Result" << endl;
    #if defined(OMP_OL)
        #pragma omp target update from(C[0:n*m])
    #elif defined(OACC)
        #pragma acc update self(C[0:n*m])
    #endif
    print_mat<et>(C, m, n);
    end_time = chrono::steady_clock::now();
    time_local["print_post-mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    //start_time = end_time;
    
    #if defined(OMP_OL)
        #pragma omp target exit data map(delete:A[0:m*k],B[0:k*n],C[0:m*n])
    #elif defined(OACC)
        #pragma acc exit data delete(A[:m*k],B[:k*n],C[0:m*n])
    #endif
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

    return 0;
}
