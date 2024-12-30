#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include "../fraction/fr.h"

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
    cout<< "----------------------------------" << endl;
}

int main(){
    
    //int n = 1000, m = 1000, k=1000;
    int n = 2, m = 2, k=2;
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
    auto time_int = test_mm<int>(12, 12, 12); cout<< "Finished MMM using 1-D array of integers" << endl; //Test 1-D array
    auto end_time = chrono::steady_clock::now();
    time_map["1d-int"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;
    auto time_fl = test_mm<float>(12, 12, 12); cout<< "Finished MMM using 1-D array of floats" << endl; //Test 1-D array
    end_time = chrono::steady_clock::now();
    time_map["1d-float"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;
    auto time_fr = test_mm_fr<fraction>(12, 12, 12); cout<< "Finished MMM using 1-D array of fractions" << endl; //Test Fraction
    end_time = chrono::steady_clock::now();
    time_map["1d-fraction"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;
    auto time_mfr = test_mm_mfr<mixed>(12, 12, 12); cout<< "Finished MMM using 1-D array of mixed fractions" << endl; //Test Fraction
    end_time = chrono::steady_clock::now();
    time_map["1d-mixed"] = chrono::duration_cast<chrono::microseconds>(end_time -start_time).count();
    start_time = end_time;

    cout<< endl << "Total Time " << endl;
    efficiency_analysis(time_map);
    cout<< endl << "1D-int Time " << endl;
    efficiency_analysis(time_int);
    cout<< endl << "1D-float Time " << endl;
    efficiency_analysis(time_fl);
    cout<< endl << "1D-fraction Time " << endl;
    efficiency_analysis(time_fr);
    cout<< endl << "1D-mixed Time " << endl;
    efficiency_analysis(time_mfr);

    return 0;
}