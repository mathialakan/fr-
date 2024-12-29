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

//Compute GCD
int gcd_(int a, int b){ int t; while(b != 0){ t = b; b = a%b; a = t; } return abs(a);}  

//User defined assert function
void assert_fr(bool expr, std::string msg){ if (!expr) std::cerr<< msg << endl;}

/*Proper and improper fractions*/
struct fraction
{
    int n; int d;
    fraction() {}
    fraction(int n, int d) : n(n), d(d) {assert_fr(d!=0, "denominator shouldn't be zero");}

    //Simplify the fraction to equivalent smallest fraction
    fraction trim(){ int gcd__ = gcd_((int)n, (int)d); return fraction(n/gcd__, d/gcd__);};
    void this_trim(){int gcd__ = gcd_((int)n, (int)d); n = n/gcd__; d = d/gcd__;};

    friend std::ostream& operator <<(std::ostream& outs, const fraction& fr) { return outs << fr.n << "/" << fr.d; };
    //fraction& operator = (const mixed &mfr){ n = mfr.fr.n + mfr.w*mfr.fr.d; d= mfr.fr.d; return *this;}

    //ALU operations between two fractions 
    inline fraction operator + (const fraction &fr) const {return (fr.d== d) ? fraction(n+fr.n, d).trim() : fraction(fr.d*n +d*fr.n, d*fr.d).trim();}
    inline fraction operator - (const fraction &fr) const {return (fr.d== d) ? fraction(n-fr.n, d).trim() : fraction(fr.d*n -d*fr.n, d*fr.d).trim();}
    inline fraction operator * (const fraction &fr) const {return fraction(n*fr.n, d*fr.d).trim();}
    inline fraction operator / (const fraction &fr) const {return fraction(n*fr.d, d*fr.n).trim();}

    inline void operator += (const fraction &fr) { if(fr.d== d)  n=n+fr.n; else {n = fr.d*n +d*fr.n; d = d*fr.d;} this_trim();} //void operator += (const fraction &fr) {fraction temp(n,d); temp = (temp +fr).trim(); n = temp.n; d = temp.d; } 
    inline void operator -= (const fraction &fr) { if(fr.d== d)  n=n-fr.n; else {n = fr.d*n -d*fr.n; d = d*fr.d;} this_trim();} //void operator -= (const fraction &fr) {fraction temp(n,d); temp = (temp -fr).trim(); n = temp.n; d = temp.d; } 
    inline void operator *= (const fraction &fr) { n = n*fr.n; d = d*fr.d; this_trim();} //void operator *= (const fraction &fr) {fraction temp(n,d); temp = (temp *fr).trim(); n = temp.n; d = temp.d; } 
    inline void operator /= (const fraction &fr) { n = n*fr.d; d = d*fr.n; this_trim();} //void operator /= (const fraction &fr) {fraction temp(n,d); temp = (temp /fr).trim(); n = temp.n; d = temp.d; } 

    inline fraction& operator++ () { n += d; return *this;}
    inline fraction operator++ (int) { fraction temp = *this; n += d; return temp;}
    inline fraction& operator-- () { n -= d; return *this;}
    inline fraction operator-- (int) { fraction temp = *this; n -= d; return temp;}

    inline bool operator == (const fraction &fr) const {return ( d == fr.d) && (n == fr.n);} // if we consider equivalent fraction - const {return (n*fr.d == fr.n*d) ;}
    inline bool operator != (const fraction &fr) const {return !( *this == fr);} //const {return ( d != fr.d) || (n != fr.n);} //const {return ( !( fraction(n,d) == fr)) ;} 
    inline bool operator > (const fraction &fr) const {return (n*fr.d > fr.n*d) ;}
    inline bool operator >= (const fraction &fr) const {return (n*fr.d >= fr.n*d) ;}
    inline bool operator < (const fraction &fr) const {return (n*fr.d < fr.n*d) ;}
    inline bool operator <= (const fraction &fr) const {return (n*fr.d <= fr.n*d) ;}
    
    //ALU operations between a fraction and a whole number 
    inline fraction operator + (const int &whole) const {return fraction(n+ whole*d, d).trim();}
    //int operator + (const fraction &whole) const {return fraction(whole*d +n, d);}  ??? - propose to include fraction to primary data type definitiations 
    inline fraction operator - (const int &whole) const {return fraction(n- whole*d, d).trim();}
    inline fraction operator * (const int &whole) const {return fraction(whole*n, d).trim();}
    inline fraction operator / (const int &whole) const {return fraction(n, whole*d).trim();}

    inline void operator += (const int &whole) { n = n+ whole*d; this_trim();} //void operator += (const int &whole) {fraction temp(n,d); temp = (temp +whole).trim(); n = temp.n; d = temp.d; } 
    inline void operator -= (const int &whole) { n = n- whole*d; this_trim();} //void operator -= (const int &whole) {fraction temp(n,d); temp = (temp -whole).trim(); n = temp.n; d = temp.d; } 
    inline void operator *= (const int &whole) { n = whole*n; this_trim();} //void operator *= (const int &whole) {fraction temp(n,d); temp = (temp *whole).trim(); n = temp.n; d = temp.d; } 
    inline void operator /= (const int &whole) { d = whole*d; this_trim();} //void operator /= (const int &whole) {fraction temp(n,d); temp = (temp /whole).trim(); n = temp.n; d = temp.d; } 

    inline bool operator == (const int &whole) const {return (n == whole*d);} // if we consider equivalent fraction - const {return (n*fr.d == fr.n*d) ;}
    inline bool operator != (const int &whole) const {return (n != whole*d);} //const {return ( d != fr.d) || (n != fr.n);} //const {return ( !( fraction(n,d) == fr)) ;} 
    inline bool operator > (const int &whole) const {return (n > whole*d) ;}
    inline bool operator >= (const int &whole) const {return (n >= whole*d) ;}
    inline bool operator < (const int &whole) const {return (n < whole*d) ;}
    inline bool operator <= (const int &whole) const {return (n <= whole*d) ;}

};


/* Mixed numbers*/
struct mixed
{
    int w; fraction fr;
    mixed() {}
    mixed(int w, fraction fr) : w(w), fr(fr) {}

    //Cast mixed number to fraction
    fraction cast_ifraction(){return (fr+w);};  //maybe improper fraction
    fraction cast_fraction(){return (fr+w).trim();}; //fraction cast_fraction(){return fraction(w*fr.d +fr.n, fr.d);};

    //Assign fraction to mixed variable - Casting from fraction to mixed (lower-cast)
    inline mixed& operator = (const fraction &fr_){ w = fr_.n/fr_.d; fr.n= abs(fr_.n%fr_.d); fr.d = fr_.d; return *this;}

    friend std::ostream& operator <<(std::ostream& outs, const mixed& mfr){return outs << mfr.w << "(" << mfr.fr.n << "/" << mfr.fr.d << ")";};

    // ALU operation between 2 mixed numbers
    inline mixed operator + (const mixed &mfr) const { mixed temp; temp =(fr +mfr.fr + w +mfr.w).trim(); return temp;}
    inline mixed operator - (const mixed &mfr) const { mixed temp; temp =(fr -mfr.fr + w -mfr.w).trim(); return temp;}
    inline mixed operator * (const mixed &mfr) const { mixed temp; temp =(fr*mfr.fr + fr*mfr.w + mfr.fr*w +mfr.w*w ).trim(); return temp;}
    inline mixed operator / ( mixed &mfr) const { mixed temp; temp =((fr +w) / mfr.cast_fraction()).trim(); return temp; }

    inline void operator += (const mixed &mfr) { mixed temp; temp = (fr +mfr.fr+ w +mfr.w).trim(); w = temp.w; fr = temp.fr;} //void operator += (const mixed &mfr) { w = w +mfr.w; fr = fr +mfr.fr;}
    inline void operator -= (const mixed &mfr) { mixed temp; temp = (fr -mfr.fr+ w -mfr.w).trim(); w = temp.w; fr = temp.fr;} //void operator -= (const mixed &mfr) { w = w -mfr.w; fr = fr -mfr.fr;}
    inline void operator *= (const mixed &mfr) { mixed temp; temp = (fr*mfr.fr + fr*mfr.w + mfr.fr*w +mfr.w*w ).trim(); w = temp.w; fr = temp.fr;}//void operator *= (const mixed &mfr) { w = w *mfr.w; fr = fraction(w*mfr.fr.n, mfr.fr.d) + fraction(mfr.w*fr.n, fr.d) + fraction(fr.n*mfr.fr.n, fr.d*mfr.fr.d);}
    inline void operator /= ( mixed &mfr) { mixed temp; temp = ((fr +w) / mfr.cast_fraction()).trim(); w = temp.w; fr = temp.fr;}

    inline mixed& operator++ () { w += 1; return *this;}
    inline mixed operator++ (int) { mixed temp = *this; w += 1; return temp;}
    inline mixed& operator-- () { w -= 1; return *this;}
    inline mixed operator-- (int) { mixed temp = *this; w -= 1; return temp;}

    inline bool operator == (const mixed &mfr) const {return (( w == mfr.w) && (fr == mfr.fr) );} // if we consider equivalent fraction - const {return ((fr+w) == mfr.cast_fraction()) ;}
    inline bool operator != ( mixed &mfr) const {return !( *this == mfr);} //const {return ( d != fr.d) || (n != fr.n);} //const {return ( !( fraction(n,d) == fr)) ;} 
    inline bool operator > ( mixed &mfr) const {return ( (fr+w) > mfr.cast_fraction()); }
    inline bool operator >= ( mixed &mfr) const {return ((fr+w) >= mfr.cast_fraction()); }
    inline bool operator < ( mixed &mfr) const {return ((fr+w) < mfr.cast_fraction()); }
    inline bool operator <= ( mixed &mfr) const {return ((fr+w) <= mfr.cast_fraction()); }

    // ALU operation between a mixed and a fraction number
    inline mixed operator + (const fraction &fr_) const {mixed temp; temp = (fr+fr_+w).trim(); return temp;}
    inline mixed operator - (const fraction &fr_) const {mixed temp; temp = (fr-fr_+w).trim(); return temp;}
    inline mixed operator * (const fraction &fr_) const {mixed temp; temp = ((fr_*w) + (fr*fr_)).trim(); return temp;}
    inline mixed operator / (const fraction &fr_) const {mixed temp; temp = ((fr+w)/fr_).trim(); return temp;}

    inline void operator += (const fraction &fr_) { mixed temp; temp =(fr+fr_+w).trim(); w = temp.w; fr = temp.fr;}
    inline void operator -= (const fraction &fr_) { mixed temp; temp =(fr-fr_+w).trim(); w = temp.w; fr = temp.fr;}
    inline void operator *= (const fraction &fr_) { mixed temp; temp =((fr_*w) + (fr*fr_)).trim(); w = temp.w; fr = temp.fr;}
    inline void operator /= (const fraction &fr_) { mixed temp; temp =((fr+w)/fr_).trim(); w = temp.w; fr = temp.fr;}

    inline bool operator == (const fraction &fr_) const {return ((fr+w) == fr_ );} // if we consider equivalent fraction - const {return ((fr+w) == mfr.cast_fraction()) ;}
    inline bool operator != ( const fraction &fr_) const {return !( *this == fr_);} //const {return ( d != fr.d) || (n != fr.n);} //const {return ( !( fraction(n,d) == fr)) ;} 
    inline bool operator > ( const fraction &fr_) const {return ((fr+w) > fr_); }
    inline bool operator >= ( const fraction &fr_) const {return ((fr+w) >= fr_); }
    inline bool operator < ( const fraction &fr_) const {return ((fr+w) < fr_); }
    inline bool operator <= ( const fraction &fr_) const {return ((fr+w) <= fr_); }

    // ALU operation between a mixed and a whole number
    inline mixed operator + (const int &whole) const {mixed temp; temp =(fr+w +whole).trim(); return temp;} //mixed operator + (const int &whole) const {return mixed(w+whole, fr);}
    inline mixed operator - (const int &whole) const {mixed temp; temp =(fr+w -whole).trim(); return temp;}
    inline mixed operator * (const int &whole) const {mixed temp; temp =((fr+w)*whole).trim(); return temp;} //mixed operator * (const int &whole) const {return mixed(w*whole, fr*whole);}
    inline mixed operator / (const int &whole) const {mixed temp; temp =((fr+w)/whole).trim(); return temp;} //mixed operator / (const int &whole) const {return cast_mixed((fraction(w,whole) +fr/whole).trim());}

    inline void operator += (const int &whole) { mixed temp; temp =(fr+w +whole).trim(); w = temp.w; fr = temp.fr;} 
    inline void operator -= (const int &whole) { mixed temp; temp =(fr+w -whole).trim(); w = temp.w; fr = temp.fr;}
    inline void operator *= (const int &whole) { mixed temp; temp =((fr+w)*whole).trim(); w = temp.w; fr = temp.fr;} 
    inline void operator /= (const int &whole) { mixed temp; temp =((fr+w)/whole).trim(); w = temp.w; fr = temp.fr;}

    inline bool operator == (const int &whole) const {return ((fr+w) == whole );} // if we consider equivalent fraction - const {return ((fr+w) == mfr.cast_fraction()) ;}
    inline bool operator != ( const int &whole) const {return !( *this == whole);} //const {return ( d != fr.d) || (n != fr.n);} //const {return ( !( fraction(n,d) == fr)) ;} 
    inline bool operator > ( const int &whole) const {return ((fr+w) > whole); }
    inline bool operator >= ( const int &whole) const {return ((fr+w) >= whole); }
    inline bool operator < ( const int &whole) const {return ((fr+w) < whole); }
    inline bool operator <= ( const int &whole) const {return ((fr+w) <= whole); }

};


mixed cast_mixed(fraction fr){ return mixed(fr.n/fr.d, fraction(abs(fr.n%fr.d), abs(fr.d))); }

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
