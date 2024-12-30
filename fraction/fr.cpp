#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <bit>
#include <cstring>
#include <cassert>
#include <variant>
#include "fr.h"

using namespace std;

/*Varient type of mixed, fraction, and whole*/
    // We need to reduce for possible small number representation: 
    // 1. mixed to fraction if whole numebr is zero
    // 2. fraction to whole if numerator is zero (Here, fraction is invalid/infinity if denominator is zero)
    // A try using c++ variant
std::variant<int, fraction, mixed> fr_variant;
////template<typename tout>
std::variant<int, fraction, mixed> reduce_ifzero(mixed mfr){ 
    if (mfr.w == 0) { 
        if (mfr.fr.n == 0) {
            fr_variant = 0; return std::get<int>(fr_variant);
            } else {
                fr_variant = mfr.fr; return std::get<fraction>(fr_variant);
                }
        } else { 
            fr_variant = mfr; return std::get<mixed> (fr_variant);  
        }
}

/*
//std::variant<int, fraction, mixed> reduce_ifzero(mixed mfr){ 
    void* reduce_ifzero(mixed mfr){ 
    std::variant<int, fraction, mixed> fr_variant;
    void* temp;
    if (mfr.w == 0) { 
        if (mfr.fr.n == 0) {
            fr_variant = 0; temp = (int*)&std::get<int>(fr_variant);
        } else {
                fr_variant = mfr.fr; temp = (fraction*)&std::get<fraction>(fr_variant);
        }
    } else { 
        fr_variant = mfr; temp = (mixed*)&std::get<mixed> (fr_variant);  
    }
    return temp;
}
*/

template<typename tout, typename tin>
tout cast(tin datain){return ( typeid(mixed) == typeid(datain)) ? datain.cast_fraction() :  mixed(datain.n/datain.d, fraction(datain.n%datain.d, datain.d));}


/*template<typename tout, typename tin>
tout cast(tin datain){
    if(typeid(tin) == typeid(mixed)) {
            if (strcmp(typeid(tout).name(), "fraction")) return fraction(datain.w*datain.fr.d +datain.fr.n, datain.fr.d);; //return datain.cast_fraction();
            //if (strcmp(typeid(tout).name(), "float")) return datain.n/datain.d; //datain.w +datain.n/datain.d;
    }else if(typeid(tin) == typeid(fraction)){
            if (strcmp(typeid(tout).name(), "mixed")) mixed(datain.n/datain.d, fraction(datain.n%datain.d, datain.d));
            //if (strcmp(typeid(tout).name(), "float")) return datain.n/datain.d;
    }
    }
    */
//tout cast(tin datain){return ( strcmp("mixed", typeid(datain).name())) ? datain.cast_fraction() :  mixed(datain.n/datain.d, fraction(datain.n%datain.d, datain.d));}

//Irrational numbers ... sqrt, etc. --- REAL WORKAROUND IS HERE!

//Test the different combination of numbers and operations
template<typename t1, typename t2>
bool test_operations(t1 a, t2 b){

    cout << a << " + " << b << " = " << (a+b) << endl;
    cout << a << " - " << b << " = " << (a-b) << endl;
    cout << a << " x " << b << " = " << (a*b) << endl;
    cout << a << " % " << b << " = " << (a/b) << endl;

    t1 c;
    c = a+b;    cout << c << " = " << a << " + " << b <<endl;    t1 d = c;    
    c += b;     cout << d << " += " << b << " , c = " << c <<endl;  d = c;    
    c -= b;     cout << d << " -= " << b << " , c = " << c <<endl;  d = c;    
    c *= b;     cout << d << " *= " << b << " , c = " << c <<endl;  d = c;    
    c /= b;     cout << d << " /= " << b << " , c = " << c <<endl;
    cout << "initial : a = " << a << " c = " << c <<endl;
    c = a++;    cout << "c=a++ : a = " << a << " c = " << c <<endl;
    c = ++a;    cout << "c=++a : a = " << a << " c = " << c <<endl;
    c = a--;    cout << "c=a-- : a = " << a << " c = " << c <<endl;
    c = --a;    cout << "c=--a : a = " << a << " c = " << c <<endl;   

    cout <<  a << " == " << b << " = " << (a==b) << endl;
    cout <<  a << " != " << b << " = " << (a!=b) << endl;
    cout <<  a << " > " << b << " = " << (a>b) << endl;
    cout <<  a << " >= " << b << " = " << (a>=b) << endl;
    cout <<  a << " < " << b << " = " << (a<b) << endl;
    cout <<  a << " <= " << b << " = " << (a<=b) << endl;

    return true;
}


int main(){

    cout<<endl;
    cout << " ----  -      -      --   ---  ---    --    -      "<<endl;
    cout << "|___  |  )   /__\\  (       |    |    |  |  | \\  |   "<<endl;
    cout << "|     |  \\  /    \\   __    |   ___    __   |  \\_|   "<<endl <<endl;

    fraction a(1, 3), b(3, 4), c(22,7), d(3,12), e(1, 0);
    mixed am(1, a), bm(2, b), cm(3, b), dm(4, d);
    mixed mt;
    mt = c;
    cout<< "mt = " << mt<<endl;
    fraction ft;
    //ft = mt;
    //cout<< "ft = " << ft<<endl;
    int x = 3;
    cout << "Testing for fraction-fraction " <<endl;
    if (test_operations(a,b)) cout << "Testing for fraction-fraction is done! " << endl << endl;
    cout << "Testing for fraction-whole " << endl ;
    if (test_operations(a,x)) cout << "Testing for fraction-whole is done! " << endl << endl;
    cout << "Testing for mixed-mixed " << endl;
    if (test_operations(am,bm)) cout << "Testing for mixed-mixed is done! " << endl << endl;
    cout << "Testing for mixed-fraction " << endl ;
    if (test_operations(am,b)) cout << "Testing for mixed-fraction is done! " << endl << endl;
    cout << "Testing for mixed-whole " << endl;
    if (test_operations(am,b)) cout << "Testing for mixed-whole is done! " << endl << endl;
    
    //cout << 3 << " + " << d << " = "<< " is equivalent to " << (3+d).trim() << endl; How to implement this? 
    cout << cm << " + " << bm << " = " << (cm+bm) << " equivalent to " << (cm+bm).cast_fraction() << " equivalent to " << (cm+bm).cast_fraction().trim() << endl;
    // cast number types between fraction, mixed, and whole ...
    //cout << cast<mixed>(a);
    //cout << cast<fraction>(am);

    mixed zero_fr(0, fraction(1,3));
    //auto trimed_zero = reduce_ifzero(zero_fr);
    //cout<< "trimed_zero : " << (int*)trimed_zero << endl;
    //*
    std::variant<int, fraction, mixed> trimed_zero = reduce_ifzero(zero_fr);
    //auto trimed_z = 0;

    try{
    if (std::holds_alternative<int>(trimed_zero)){
        auto trimed_z = std::get<int>(trimed_zero);
        cout << zero_fr << " = " << std::get<int>(trimed_zero) << endl;
    }
    else if (std::holds_alternative<fraction>(trimed_zero)){
        auto trimed_z = std::get<fraction>(trimed_zero);
        cout << zero_fr << " = " << std::get<fraction>(trimed_zero) << endl;
    } else if (std::holds_alternative<mixed>(trimed_zero)){
         cout << zero_fr << " = " << std::get<mixed>(trimed_zero) << endl;
    }
    } catch(std::bad_variant_access e){
        std::cerr << "Error: " << e.what() <<endl;
    }
   // */
}
