/* This is a program for testing floating point effect
It uses pi in varying precisions and computed volume of a given sphare */
/* 
1. float: single-precision floatin point number ~ 7 decimal digits of precision. 32 bits of memory
2. double: double-precision floatin point number ~ 15 decimal digits of precision. 64 bits of memory
*/
#include <iostream>
#include <iomanip>
#include "../fraction/fr.h"
//#include <floatx>
using namespace std;

// C/C++ (21 digits) M_PI 3.14159265358979323846
// Go (63 digits)      Pi 3.14159265358979323846264338327950288419716939937510582097494459

#define pi_e15m112 3.141592653589793238462643383279503  // 34 decimal points
#define pi_e11m52 3.141592653589793                     // 15 decimal points
#define pi_e8m23 3.141593                               // 7 decimnal points
#define pi_e5m10 3.142                                  // 3 decimal points
#define pi_e4m3 3                                       // 0 decimal points
#define pi_e5m2 3                                       // 0 decimal points

//typedef struct{ float n; float d;}FR;

const double pi_d = 3.141592653589793; //23846 ;
const float pi_f = 3.1415927;

template<typename T_r, typename T_pi>
auto cal_volume(T_r radius, T_pi pi_){
    return (4/3)*pi_*radius*radius*radius;
}

template<typename T_r>
auto cal_volume_fr(T_r radius, fraction pi_){
    return  fraction(4,3)*pi_*radius*radius*radius;
}

template<typename T_fr, typename T_fp>
auto comp_accuracy(T_fr val_fr, T_fp val_fp){
    return (val_fr - abs(val_fp -val_fr))*100/val_fr ;
}

template<typename T_r, typename T_pi>
void test_encoding(T_r radius, T_pi pi_, T_r exact ){
    auto vol = cal_volume(radius, pi_);
    //cout << std::setprecision(34);
    cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
    cout<< "Volume of the sphare of radius " << radius << " using PI = " << pi_ << " is " ;
    cout << std::setprecision(10); cout << vol  ;
    cout << std::setprecision(5); cout << " and  Accuracy: " << comp_accuracy(exact, vol) << " % "<< endl;
}

int main(){
    //FR pi_fr = {22, 7};
    fraction pi_fr(22,7);
    double r = 21;

    auto vol_fr = cal_volume_fr(21, pi_fr);
    double vol_double = cast_double(vol_fr);
    cout<< "Volume of the sphare of radius " << r << " using PI = " << pi_fr << " is " << vol_fr << " in FP : " << vol_double <<  endl;

    test_encoding<double>(21, pi_e15m112, vol_double);
    test_encoding<double>(21, pi_e11m52, vol_double);
    test_encoding<double>(21, pi_e8m23, vol_double);
    test_encoding<double>(21, pi_e5m10, vol_double);
    test_encoding<double>(21, pi_e4m3, vol_double);
    test_encoding<double>(21, pi_e5m2, vol_double);
    test_encoding<double>(21, M_PI, vol_double);

    return 0;
}
