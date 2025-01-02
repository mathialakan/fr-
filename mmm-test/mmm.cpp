#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include "../fraction/fr.h"


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
bool fill_mat(et* A, int N, int M){
    srand(time(0));

    for(int i=0; i <N; ++i)
    for(int j=0; j <M; ++j)
        A[M*i +j] = 1 + (rand()% EMAX);

    return true;
}

//template<typename et>
bool fill_mat_fr(fraction* A, int N, int M){
    srand(time(0));

    for(int i=0; i <N; ++i){
    for(int j=0; j <M; ++j){
        fraction e(1 + (rand()% NMAX), 1 + (rand()% DMAX));
        A[M*i +j] = e;
    }
    }
    return true;
}

bool fill_mat_mfr(mixed* A, int N, int M){
    srand(time(0));

    for(int i=0; i <N; ++i){
    for(int j=0; j <M; ++j){
        mixed e( 1 + (rand()% WMAX), fraction(1 + (rand()% NMAX), 1 + (rand()% DMAX)));
        A[M*i +j] = e;
    }
    }
    return true;
}

template<typename et>
et* mat_mul(et* A, et* B, int N, int K, int M){

    et* C = (et*)malloc(N*M *sizeof(et));
    #if defined(OMP_OL)
        #pragma omp target enter data map(alloc: C[:n*m])
    #elif defined(OACC)
        #pragma acc enter data create(C[:n*m])
    #endif

    #if defined(OMP)
        #pragma omp parallel for collapse(2) schedule(dynamic)
    #elif defined(OMP_OL)
        #pragma omp target teams distributed parallel for collapse(2) \
            thread_limit(team_size) num_teams((N*M +1)/team_size) 
    #elif defined(OACC)
        #pragma acc parallel loop present(A, B, C) gang worker \
                num_workers(team_size) vector_length(32)
    #endif
    for(int i=0; i<N; i++)
        for(int j=0; j<M; j++){
            C[M*i +j]= 0;
            for(int k=0; k<K; k++)
                C[M*i +j] += A[K*i + k] *B[M*k +j]; 
        }
    return C;
}

template<typename et>
bool print_mat(et* A, int N, int M){
    for(int i=0; i<N; i++){
        for(int j=0; j<M; j++)
            cout<< "\t" <<A[M*i+j];
        cout <<endl;
    }
    return true;
}

/* Using c-style pointer 2D arrays */

template<typename et>
bool fill_mat(et** A){

    int N = sizeof(A)/sizeof(A[0]);
    int M = sizeof(A[0])/sizeof(A[0][0]);

    for(int i=0; i <N; ++i)
    for(int j=0; j <M; ++j)
        A[i][j] = 1+ (rand()%EMAX);
    
    return true;
}
template<typename et>
et** mat_maul(et** A, et** B){

    //int size_bytes = sizeof(A);
    int N = sizeof(A)/sizeof(A[0]);
    int K = sizeof(A[0])/sizeof(A[0][0]);
    int M = sizeof(B)/sizeof(B[0])/sizeof(B[0][0]);

    et** C = (et**) malloc(N*sizeof(et*));
    for(int i=0; i<N; i++)
        C[i] = (et*)malloc(M*sizeof(et));

    #if defined(OMP_OL)
        #pragma omp target enter data map(alloc: C[:n])
    #elif defined(OACC)
        #pragma acc enter data create(C[:n])
    #endif

    #if defined(OMP)
        #pragma omp parallel for collapse(2) schedule(dynamic) 
    #elif defined(OMP_OL)
        #pragma omp target teams distribute parallel for \
                thread_limit(team_size) num_teams((N*M +1)/team_size)
    #elif defined(OACC)
        #pragma acc parallel loop
    #endif
    for(int i=0; i<N; i++)
        for(int j=0; j<M; j++){
            C[i][j] = 0;
            for(int k=0; k<K; k++)
                C[i][j] += A[i][k]*B[k][j];
        }
    return C;        

}

/* Using cpp-style 1D pointer arrays */
template<typename et>
et* mat_mul_cpp(et* A, et* B, int N, int K, int M){

    et* C = new et[N*M];
    
    #if defined(OMP_OL)
        #pragma omp target enter data map(alloc: C[:n*m])
    #elif defined(OACC)
        #pragma acc enter data create(C[:n*m])
    #endif

    #if defined(OMP)
        #pragma omp parallel for collapse(2) schedule(dynamic)
    #elif defined(OMP_OL)
        #pragma omp taraget teams distribute parallel for \
                thread_limit(team_size) num_teams((N*M +1)/team_size)
    #elif defined(OACC)
        #pragma acc parallel loop
    #endif
    for(int i=0; i<N; i++)
    for(int j=0; j<M; j++){
        C[M*(i-1) +j] = 0;
        for(int k=0; k<K; k++)
            C[M*(i-1) +j] += A[K*(i-1)+k]*B[M*(k-1)+j];
    }
    return C;

}

/* Using cpp-style 2D pointer arrays */
template<typename et>
et** mat_maul_cpp(et** A, et** B){

    int N = sizeof(A)/sizeof(A[0]);
    int K = sizeof(A[0])/sizeof(A[0][0]);
    int M = sizeof(B)/sizeof(B[0])/sizeof(B[0][0]);

    et** C =  new et*[N];
    for (int i=0; i<N; i++)
        C[i] = new et[M];

    #if defined(OMP_OL)
        #pragma omp target enter data map(alloc: C[:n])
    #elif defined(OACC)
        #pragma acc enter data create(C[:n])
    #endif

    #if defined(OMP)
        #pragma omp parallel for collapse(2) schedule(dynamic)
    #elif defined(OMP_OL)
        #pragma omp taraget teams distribute parallel for \
                thread_limit(team_size) num_teams((N*M +1)/team_size)
    #elif defined(OACC)
        #pragma acc parallel loop
    #endif
    for (int i=0; i<N; i++)
        for( int j=0; j<M; j++){
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
    int N = A.size();
    int K = A[0].size();
    int M = B[0].size();
    //if (K != B.size()) return 0;
    vector<vector<et> > C(N, vector<et>(M));

    #if defined(OMP)
        #pragma omp parallel for collapse(2) schedule(dynamic)
    #elif defined(OMP_OL)
        #pragma omp taraget teams distribute parallel for \
                thread_limit(team_size) num_teams((N*M +1)/team_size)
    #elif defined(OACC)
        #pragma acc parallel loop
    #endif  
    for(int i=0; i< N; ++i)
        for(int j=0; j< M; ++j){
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
unordered_map<string, double> test_mm(int n, int k, int m){

    unordered_map<string, double> time_local;

    et* A = (et*)malloc( n*k* sizeof(et));
    et* B = (et*)malloc( k*m* sizeof(et));
    auto start_time = chrono::steady_clock::now();

    fill_mat(A, n, k);
    fill_mat(B, k, m);
    #if defined(OMP_OL)
        #prgama omp target enter data map(to:A[0:n*k],B[0:k*m])
    #elif defined(OACC)
        #pragma acc enter data copyin(A[:n*k],B[:k*m])
    #endif
    auto end_time = chrono::steady_clock::now();
    time_local["fill_mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    print_mat<et>(A, n, k);
    cout<<endl;
    print_mat<et>(B, k, m);
    end_time = chrono::steady_clock::now();
    time_local["print_pre-mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    auto C = mat_mul<et>(A, B, n, k, m);
    end_time = chrono::steady_clock::now();
    time_local["comp_mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    cout << "Result" << endl;
    #if defined(OMP_OL)
        #prgama omp target update from(C[0:n*m])
    #elif defined(OACC)
        #pragma acc update self(C[0:n*m])
    #endif
    print_mat<et>(C, n, m);
    end_time = chrono::steady_clock::now();
    time_local["print_post-mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    //start_time = end_time;

    #if defined(OMP_OL)
        #prgama omp target exit data map(delete:A[0:n*k],B[0:k*m],C[0:m*n])
    #elif defined(OACC)
        #pragma acc exit data delete(A[:n*k],B[:k*m],C[0:m*n])
    #endif

    return time_local;
}

template<typename et>
unordered_map<string, double> test_mm_fr(int n, int k, int m){

    unordered_map<string, double> time_local;

    et* A = (et*)malloc( n*k* sizeof(et));
    et* B = (et*)malloc( k*m* sizeof(et));
    auto start_time = chrono::steady_clock::now();

    fill_mat_fr(A, n, k);
    fill_mat_fr(B, k, m); 
    auto end_time = chrono::steady_clock::now();
    time_local["fill_mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    print_mat<et>(A, n, k);
    cout<<endl;
    print_mat<et>(B, k, m);
    end_time = chrono::steady_clock::now();
    time_local["print_pre-mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    auto C = mat_mul<et>(A, B, n, k, m);
    end_time = chrono::steady_clock::now();
    time_local["comp_mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    cout << "Result" << endl;
    print_mat<et>(C, n, m);
    end_time = chrono::steady_clock::now();
    time_local["print_post-mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    //start_time = end_time;

    return time_local;
}

template<typename et>
unordered_map<string, double> test_mm_mfr(int n, int k, int m){

    unordered_map<string, double> time_local;

    et* A = (et*)malloc( n*k* sizeof(et));
    et* B = (et*)malloc( k*m* sizeof(et));
    auto start_time = chrono::steady_clock::now();
    fill_mat_mfr(A, n, k);
    fill_mat_mfr(B, k, m); 
    auto end_time = chrono::steady_clock::now();
    time_local["fill_mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;
    
    print_mat<et>(A, n, k);
    cout<<endl;
    print_mat<et>(B, k, m);
    end_time = chrono::steady_clock::now();
    time_local["print_pre-mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    auto C = mat_mul<et>(A, B, n, k, m);
    end_time = chrono::steady_clock::now();
    time_local["comp_mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    cout << "Result" << endl;
    print_mat<et>(C, n, m);
    end_time = chrono::steady_clock::now();
    time_local["print_post-mat"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    //start_time = end_time;
    
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

void efficiency_analysis(unordered_map<string, double> time_map , size_t dsize, size_t tsize){

    cout<< "Data size (in Bytes) : " <<dsize<<"\t Type size (in Bytes) : " << tsize  << endl;
    cout<< "----------------------------------" << endl;
    cout<< "Task" << "\t" << "Time (micro-sec)" << endl;
    cout<< "----------------------------------" << endl;
    for(auto& pair : time_map)
        cout<< pair.first << "\t" << pair.second << endl;
    cout<< "----------------------------------" << endl <<endl;

    cout<< "----------------------------------" << endl;
    cout<< "Task" << "\t" << "Bandwidth (GB/s)" << endl;
    cout<< "----------------------------------" << endl;
    //auto seconds = 0;
    //auto gigabytes = 0;
    for(auto& pair : time_map){
        auto seconds = pair.second * 1.e-6;
        auto gigabytes = (double)(dsize*tsize)* 1.e-9; // GB
        cout<< pair.first << "\t" << (gigabytes/seconds) << endl;
    }       
    cout<< "----------------------------------" << endl;
}

int main(){
    
    int n = 100, m = 100, k=100;
    //int n = 2, m = 2, k=2;
    // 2D-Vector of floats
    /*
    vector<vector<float> > A(n, vector<float>(k));
    vector<vector<float> > B(k, vector<float>(m));

    //if (read_mat(A) == 1) cout<< "Reading is failed" << endl;
    //if (read_mat(B) == 1) cout<< "Reading is failed" << endl;
    fill_mat(A);
    fill_mat(B);

    auto C = mat_mul(A,B);
    cout << "Result" << endl;
    print_mat((vector<vector<float> >)C);
    */

    unordered_map<string, double> time_map;
    auto start_time = chrono::steady_clock::now();
    auto time_int = test_mm<int>(n, k, m); cout<< "Finished MMM using 1-D array of integers" << endl; //Test 1-D array
    auto end_time = chrono::steady_clock::now();
    time_map["1d-int"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;
    auto time_fl = test_mm<float>(n, k, m); cout<< "Finished MMM using 1-D array of floats" << endl; //Test 1-D array
    end_time = chrono::steady_clock::now();
    time_map["1d-float"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;
    auto time_fr = test_mm_fr<fraction>(n, k, m); cout<< "Finished MMM using 1-D array of fractions" << endl; //Test Fraction
    end_time = chrono::steady_clock::now();
    time_map["1d-fraction"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;
    auto time_mfr = test_mm_mfr<mixed>(n, k, m); cout<< "Finished MMM using 1-D array of mixed fractions" << endl; //Test Fraction
    end_time = chrono::steady_clock::now();
    time_map["1d-mixed"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    size_t dsize = n*k +k*m +2*n*m;
    cout<< endl << "Total Time " << endl;
    efficiency_analysis(time_map);
    cout<< endl << "1D-int Time " << endl;
    efficiency_analysis(time_int, dsize, sizeof(int)); 
    cout<< endl << "1D-float Time " << endl;
    efficiency_analysis(time_fl, dsize, sizeof(float));
    cout<< endl << "1D-fraction Time " << endl;
    efficiency_analysis(time_fr, dsize, sizeof(fraction));
    cout<< endl << "1D-mixed Time " << endl;
    efficiency_analysis(time_mfr, dsize, sizeof(mixed));

    
    return 0;
}