#ifndef FR_H
#define FR_H

#include <iostream>
#include <cassert>
//Compute GCD
#if defined(GPU)
__host__ __device__
#endif
int gcd_(int a, int b){ int t; while(b != 0){ t = b; b = a%b; a = t; } return abs(a);}  

//User defined assert function
//void assert_fr(bool expr, std::string msg){ if (!expr) std::cerr<< msg;}

/*Proper and improper fractions  - 8 bytes (Current representation uses int as a base type (2 ints = 8 bytes) --- we need to consider proper base data-type to reduce the size)*/
struct fraction
{
    int n; int d;
#if defined(GPU)
__host__ __device__
#endif
    fraction() {}
#if defined(GPU)
__host__ __device__
#endif
    fraction(int n, int d) : n(n), d(d) {assert(d!=0);}
    //fraction(int n, int d) : n(n), d(d) {assert_fr(d!=0, "denominator shouldn't be zero");}

    //Simplify the fraction to equivalent smallest fraction
#if defined(GPU)
__host__ __device__
#endif
    fraction trim(){ int gcd__ = gcd_((int)n, (int)d); return fraction(n/gcd__, d/gcd__);};
#if defined(GPU)
__host__ __device__
#endif
    void this_trim(){int gcd__ = gcd_((int)n, (int)d); n = n/gcd__; d = d/gcd__;};
    friend std::ostream& operator <<(std::ostream& outs, const fraction& fr) { return outs << fr.n << "/" << fr.d; };
    //fraction& operator = (const mixed &mfr){ n = mfr.fr.n + mfr.w*mfr.fr.d; d= mfr.fr.d; return *this;}

    //Assign whole to fraction variable - Casting from whole to fraction
    //Assignment doesn't work for cuda/hip code
    //__host__ __device__ inline fraction& operator = (const int& whole){ n=whole; d=1; return *this;}
#if defined(GPU)
__host__ __device__
#endif
    inline fraction& operator = (const int& whole){ n=whole; d=1; return *this;}
    //inline fraction& operator = ( fraction &fr){ this->n =fr.n; this->d = fr.d; this->trim(); return *this;}
    //For the device codes (cuda, hip, etc.)
    //__device__  fraction assign_int(const int& whole){ n=whole; d=1; return *this;}

    //ALU operations between two fractions 
    //inline fraction operator + (const fraction &fr) const {return (fr.d== d) ? fraction(n+fr.n, d).trim() : fraction(fr.d*n +d*fr.n, d*fr.d).trim();} // 4 ops
    //inline fraction operator - (const fraction &fr) const {return (fr.d== d) ? fraction(n-fr.n, d).trim() : fraction(fr.d*n -d*fr.n, d*fr.d).trim();} // 4 ops
#if defined(GPU)
__host__ __device__
#endif
    inline fraction operator + (const fraction &fr) const {return fraction(fr.d*n +d*fr.n, d*fr.d).trim();} // 7 ops = 4 ops + atleast 3 ops (relational, modulus, and absolute) by trim
#if defined(GPU)
__host__ __device__
#endif
    inline fraction operator - (const fraction &fr) const {return fraction(fr.d*n -d*fr.n, d*fr.d).trim();} // 7 ops = 4 ops + atleast 3 ops by trim
#if defined(GPU)
__host__ __device__
#endif
    inline fraction operator * (const fraction &fr) const {return fraction(n*fr.n, d*fr.d).trim();}  // 5 ops = 2 ops + atleast 3 ops by trim
#if defined(GPU)
__host__ __device__
#endif
    inline fraction operator / (const fraction &fr) const {return fraction(n*fr.d, d*fr.n).trim();}  // 5 ops = 2 ops + atleast 3 ops by trim

    //inline void operator += (const fraction &fr) { if(fr.d== d)  n=n+fr.n; else {n = fr.d*n +d*fr.n; d = d*fr.d;} this_trim();} //void operator += (const fraction &fr) {fraction temp(n,d); temp = (temp +fr).trim(); n = temp.n; d = temp.d; } 
    //inline void operator -= (const fraction &fr) { if(fr.d== d)  n=n-fr.n; else {n = fr.d*n -d*fr.n; d = d*fr.d;} this_trim();} //void operator -= (const fraction &fr) {fraction temp(n,d); temp = (temp -fr).trim(); n = temp.n; d = temp.d; } 
#if defined(GPU)
__host__ __device__
#endif
    inline void operator += (const fraction &fr) { n = fr.d*n +d*fr.n; d = d*fr.d; this_trim();} //void operator += (const fraction &fr) {fraction temp(n,d); temp = (temp +fr).trim(); n = temp.n; d = temp.d; } 
#if defined(GPU)
__host__ __device__
#endif
    inline void operator -= (const fraction &fr) { n = fr.d*n -d*fr.n; d = d*fr.d; this_trim();} //void operator -= (const fraction &fr) {fraction temp(n,d); temp = (temp -fr).trim(); n = temp.n; d = temp.d; } 
#if defined(GPU)
__host__ __device__
#endif
    inline void operator *= (const fraction &fr) { n = n*fr.n; d = d*fr.d; this_trim();} //void operator *= (const fraction &fr) {fraction temp(n,d); temp = (temp *fr).trim(); n = temp.n; d = temp.d; } 
#if defined(GPU)
__host__ __device__
#endif
    inline void operator /= (const fraction &fr) { n = n*fr.d; d = d*fr.n; this_trim();} //void operator /= (const fraction &fr) {fraction temp(n,d); temp = (temp /fr).trim(); n = temp.n; d = temp.d; } 

#if defined(GPU)
__host__ __device__
#endif
    inline fraction& operator++ () { n += d; return *this;}
#if defined(GPU)
__host__ __device__
#endif
    inline fraction operator++ (int) { fraction temp = *this; n += d; return temp;}
#if defined(GPU)
__host__ __device__
#endif
    inline fraction& operator-- () { n -= d; return *this;}
#if defined(GPU)
__host__ __device__
#endif
    inline fraction operator-- (int) { fraction temp = *this; n -= d; return temp;}

#if defined(GPU)
__host__ __device__
#endif
    inline bool operator == (const fraction &fr) const {return ( d == fr.d) && (n == fr.n);} // if we consider equivalent fraction - const {return (n*fr.d == fr.n*d) ;}
#if defined(GPU)
__host__ __device__
#endif
    inline bool operator != (const fraction &fr) const {return !( *this == fr);} //const {return ( d != fr.d) || (n != fr.n);} //const {return ( !( fraction(n,d) == fr)) ;} 
#if defined(GPU)
__host__ __device__
#endif
    inline bool operator > (const fraction &fr) const {return (n*fr.d > fr.n*d) ;}
#if defined(GPU)
__host__ __device__
#endif
    inline bool operator >= (const fraction &fr) const {return (n*fr.d >= fr.n*d) ;}
#if defined(GPU)
__host__ __device__
#endif
    inline bool operator < (const fraction &fr) const {return (n*fr.d < fr.n*d) ;}
#if defined(GPU)
__host__ __device__
#endif    
    inline bool operator <= (const fraction &fr) const {return (n*fr.d <= fr.n*d) ;}
    
    //ALU operations between a fraction and a whole number 
#if defined(GPU)
__host__ __device__
#endif
    inline fraction operator + (const int &whole) const {return fraction(n+ whole*d, d).trim();} // 5 ops = 2 ops + 3 ops 
    //int operator + (const fraction &whole) const {return fraction(whole*d +n, d);}  ??? - propose to include fraction to primary data type definitiations 
#if defined(GPU)
__host__ __device__
#endif
    inline fraction operator - (const int &whole) const {return fraction(n- whole*d, d).trim();} // 5 ops =2 ops + 3 ops 
#if defined(GPU)
__host__ __device__
#endif
    inline fraction operator * (const int &whole) const {return fraction(whole*n, d).trim();} // 4 ops = 1 ops + 3 ops 
#if defined(GPU)
__host__ __device__
#endif
    inline fraction operator / (const int &whole) const {return fraction(n, whole*d).trim();} // 4 ops = 1 ops + 3 ops 

#if defined(GPU)
__host__ __device__
#endif
    inline void operator += (const int &whole) { n = n+ whole*d; this_trim();} //void operator += (const int &whole) {fraction temp(n,d); temp = (temp +whole).trim(); n = temp.n; d = temp.d; } 
#if defined(GPU)
__host__ __device__
#endif
    inline void operator -= (const int &whole) { n = n- whole*d; this_trim();} //void operator -= (const int &whole) {fraction temp(n,d); temp = (temp -whole).trim(); n = temp.n; d = temp.d; } 
#if defined(GPU)
__host__ __device__
#endif
    inline void operator *= (const int &whole) { n = whole*n; this_trim();} //void operator *= (const int &whole) {fraction temp(n,d); temp = (temp *whole).trim(); n = temp.n; d = temp.d; } 
#if defined(GPU)
__host__ __device__
#endif
    inline void operator /= (const int &whole) { d = whole*d; this_trim();} //void operator /= (const int &whole) {fraction temp(n,d); temp = (temp /whole).trim(); n = temp.n; d = temp.d; } 
#if defined(GPU)
__host__ __device__
#endif
    inline bool operator == (const int &whole) const {return (n == whole*d);} // if we consider equivalent fraction - const {return (n*fr.d == fr.n*d) ;}
#if defined(GPU)
__host__ __device__
#endif    
    inline bool operator != (const int &whole) const {return (n != whole*d);} //const {return ( d != fr.d) || (n != fr.n);} //const {return ( !( fraction(n,d) == fr)) ;} 
#if defined(GPU)
__host__ __device__
#endif
    inline bool operator > (const int &whole) const {return (n > whole*d) ;}
#if defined(GPU)
__host__ __device__
#endif
    inline bool operator >= (const int &whole) const {return (n >= whole*d) ;}
#if defined(GPU)
__host__ __device__
#endif
    inline bool operator < (const int &whole) const {return (n < whole*d) ;}
#if defined(GPU)
__host__ __device__
#endif
    inline bool operator <= (const int &whole) const {return (n <= whole*d) ;}

};


/* Mixed numbers  - 12 bytes (Current representation uses int as a base type (3 ints = 12 bytes) --- we need to consider proper base data-type to reduce the size)*/ 
struct mixed
{
    int w; fraction fr;
#if defined(GPU)
__host__ __device__
#endif
    mixed() {}
#if defined(GPU)
__host__ __device__
#endif
    mixed(int w, fraction fr) : w(w), fr(fr) {}

    //Cast mixed number to fraction
#if defined(GPU)
__host__ __device__
#endif
    fraction cast_ifraction(){return (fr+w);};  //maybe improper fraction
#if defined(GPU)
__host__ __device__
#endif
    fraction cast_fraction(){return (fr+w).trim();}; //fraction cast_fraction(){return fraction(w*fr.d +fr.n, fr.d);};

    //Assign fraction to mixed variable - Casting from fraction to mixed (lower-cast)
#if defined(GPU)
__host__ __device__
#endif
    inline mixed& operator = (const fraction &fr_){ w = fr_.n/fr_.d; fr.n= abs(fr_.n%fr_.d); fr.d = fr_.d; return *this;} // 3
    //Assign whole to mixed variable - Casting from whole to mixed
    //__device__ __host__ inline mixed& operator = (const int &whole){ w=whole; fr.n=0; fr.d=1; return *this;}
#if defined(GPU)
__host__ __device__
#endif
    inline mixed& operator = (const int &whole){ w=whole; fr.n=0; fr.d=1; return *this;}

    friend std::ostream& operator <<(std::ostream& outs, const mixed& mfr){return outs << mfr.w << "(" << mfr.fr.n << "/" << mfr.fr.d << ")";};

    // ALU operation between 2 mixed numbers
#if defined(GPU)
__host__ __device__
#endif
    inline mixed operator + (const mixed &mfr) const { mixed temp; temp =(fr +mfr.fr + w +mfr.w).trim(); return temp;}  // 23 ops = 7 + 5 + 5 +3(a) +3(trim)
#if defined(GPU)
__host__ __device__
#endif
    inline mixed operator - (const mixed &mfr) const { mixed temp; temp =(fr -mfr.fr + w -mfr.w).trim(); return temp;}  // 23 ops = 7 + 5 + 5 +3(a) +3(trim)
#if defined(GPU)
__host__ __device__
#endif    
    inline mixed operator * (const mixed &mfr) const { mixed temp; temp =(fr*mfr.fr + fr*mfr.w + mfr.fr*w +mfr.w*w ).trim(); return temp;}  // 20 ops = (5 + 4 + 4 + 1) +3(a) +3(trim)
#if defined(GPU)
__host__ __device__
#endif
    inline mixed operator / ( mixed &mfr) const { mixed temp; temp =((fr +w) / mfr.cast_fraction()).trim(); return temp; }  // 24 ops  = 5 + 5 + (5 +3)  +3(a) +3(trim)
#if defined(GPU)
__host__ __device__
#endif
    inline void operator += (const mixed &mfr) { mixed temp; temp = (fr +mfr.fr+ w +mfr.w).trim(); w = temp.w; fr = temp.fr;} //void operator += (const mixed &mfr) { w = w +mfr.w; fr = fr +mfr.fr;}
#if defined(GPU)
__host__ __device__
#endif
    inline void operator -= (const mixed &mfr) { mixed temp; temp = (fr -mfr.fr+ w -mfr.w).trim(); w = temp.w; fr = temp.fr;} //void operator -= (const mixed &mfr) { w = w -mfr.w; fr = fr -mfr.fr;}
#if defined(GPU)
__host__ __device__
#endif
    inline void operator *= (const mixed &mfr) { mixed temp; temp = (fr*mfr.fr + fr*mfr.w + mfr.fr*w +mfr.w*w ).trim(); w = temp.w; fr = temp.fr;}//void operator *= (const mixed &mfr) { w = w *mfr.w; fr = fraction(w*mfr.fr.n, mfr.fr.d) + fraction(mfr.w*fr.n, fr.d) + fraction(fr.n*mfr.fr.n, fr.d*mfr.fr.d);}
#if defined(GPU)
__host__ __device__
#endif
    inline void operator /= ( mixed &mfr) { mixed temp; temp = ((fr +w) / mfr.cast_fraction()).trim(); w = temp.w; fr = temp.fr;}
#if defined(GPU)
__host__ __device__
#endif
    inline mixed& operator++ () { w += 1; return *this;}
#if defined(GPU)
__host__ __device__
#endif
    inline mixed operator++ (int) { mixed temp = *this; w += 1; return temp;}
#if defined(GPU)
__host__ __device__
#endif
    inline mixed& operator-- () { w -= 1; return *this;}
#if defined(GPU)
__host__ __device__
#endif
    inline mixed operator-- (int) { mixed temp = *this; w -= 1; return temp;}

#if defined(GPU)
__host__ __device__
#endif
    inline bool operator == (const mixed &mfr) const {return (( w == mfr.w) && (fr == mfr.fr) );} // if we consider equivalent fraction - const {return ((fr+w) == mfr.cast_fraction()) ;}
#if defined(GPU)
__host__ __device__
#endif
    inline bool operator != ( mixed &mfr) const {return !( *this == mfr);} //const {return ( d != fr.d) || (n != fr.n);} //const {return ( !( fraction(n,d) == fr)) ;} 
#if defined(GPU)
__host__ __device__
#endif
    inline bool operator > ( mixed &mfr) const {return ( (fr+w) > mfr.cast_fraction()); }
#if defined(GPU)
__host__ __device__
#endif
    inline bool operator >= ( mixed &mfr) const {return ((fr+w) >= mfr.cast_fraction()); }
#if defined(GPU)
__host__ __device__
#endif
    inline bool operator < ( mixed &mfr) const {return ((fr+w) < mfr.cast_fraction()); }
#if defined(GPU)
__host__ __device__
#endif
    inline bool operator <= ( mixed &mfr) const {return ((fr+w) <= mfr.cast_fraction()); }

    // ALU operation between a mixed and a fraction number
#if defined(GPU)
__host__ __device__
#endif
    inline mixed operator + (const fraction &fr_) const {mixed temp; temp = (fr+fr_+w).trim(); return temp;}  // 18 ops = 7 +5 +3(a) +3(trim)
#if defined(GPU)
__host__ __device__
#endif
    inline mixed operator - (const fraction &fr_) const {mixed temp; temp = (fr-fr_+w).trim(); return temp;}  // 18 ops = 7 +5 +3(a) +3(trim)
#if defined(GPU)
__host__ __device__
#endif
    inline mixed operator * (const fraction &fr_) const {mixed temp; temp = ((fr_*w) + (fr*fr_)).trim(); return temp;} // 15 ops = 4 +5 +3(a) +3(trim)
#if defined(GPU)
__host__ __device__
#endif
    inline mixed operator / (const fraction &fr_) const {mixed temp; temp = ((fr+w)/fr_).trim(); return temp;} // 16 ops = 5 + 5 +3(a) +3(trim)

#if defined(GPU)
__host__ __device__
#endif
    inline void operator += (const fraction &fr_) { mixed temp; temp =(fr+fr_+w).trim(); w = temp.w; fr = temp.fr;}
#if defined(GPU)
__host__ __device__
#endif
    inline void operator -= (const fraction &fr_) { mixed temp; temp =(fr-fr_+w).trim(); w = temp.w; fr = temp.fr;}
#if defined(GPU)
__host__ __device__
#endif
    inline void operator *= (const fraction &fr_) { mixed temp; temp =((fr_*w) + (fr*fr_)).trim(); w = temp.w; fr = temp.fr;}
#if defined(GPU)
__host__ __device__
#endif
    inline void operator /= (const fraction &fr_) { mixed temp; temp =((fr+w)/fr_).trim(); w = temp.w; fr = temp.fr;}

#if defined(GPU)
__host__ __device__
#endif
    inline bool operator == (const fraction &fr_) const {return ((fr+w) == fr_ );} // if we consider equivalent fraction - const {return ((fr+w) == mfr.cast_fraction()) ;}
#if defined(GPU)
__host__ __device__
#endif
    inline bool operator != ( const fraction &fr_) const {return !( *this == fr_);} //const {return ( d != fr.d) || (n != fr.n);} //const {return ( !( fraction(n,d) == fr)) ;} 
#if defined(GPU)
__host__ __device__
#endif
    inline bool operator > ( const fraction &fr_) const {return ((fr+w) > fr_); }
#if defined(GPU)
__host__ __device__
#endif
    inline bool operator >= ( const fraction &fr_) const {return ((fr+w) >= fr_); }
#if defined(GPU)
__host__ __device__
#endif
    inline bool operator < ( const fraction &fr_) const {return ((fr+w) < fr_); }
#if defined(GPU)
__host__ __device__
#endif
    inline bool operator <= ( const fraction &fr_) const {return ((fr+w) <= fr_); }

    // ALU operation between a mixed and a whole number
#if defined(GPU)
__host__ __device__
#endif
    inline mixed operator + (const int &whole) const {mixed temp; temp =(fr+w +whole).trim(); return temp;}  // 16 ops = 5 +5 +3  +3 //mixed operator + (const int &whole) const {return mixed(w+whole, fr);}
#if defined(GPU)
__host__ __device__
#endif
    inline mixed operator - (const int &whole) const {mixed temp; temp =(fr+w -whole).trim(); return temp;}  // 16 ops = 5 +5 +3  +3
#if defined(GPU)
__host__ __device__
#endif
    inline mixed operator * (const int &whole) const {mixed temp; temp =((fr+w)*whole).trim(); return temp;} // 15 ops = 5 +4 +3  +3 //mixed operator * (const int &whole) const {return mixed(w*whole, fr*whole);}
#if defined(GPU)
__host__ __device__
#endif
    inline mixed operator / (const int &whole) const {mixed temp; temp =((fr+w)/whole).trim(); return temp;} // 15 ops = 5 +4 +3  +3 //mixed operator / (const int &whole) const {return cast_mixed((fraction(w,whole) +fr/whole).trim());}

#if defined(GPU)
__host__ __device__
#endif
    inline void operator += (const int &whole) { mixed temp; temp =(fr+w +whole).trim(); w = temp.w; fr = temp.fr;} 
#if defined(GPU)
__host__ __device__
#endif
    inline void operator -= (const int &whole) { mixed temp; temp =(fr+w -whole).trim(); w = temp.w; fr = temp.fr;}
#if defined(GPU)
__host__ __device__
#endif
    inline void operator *= (const int &whole) { mixed temp; temp =((fr+w)*whole).trim(); w = temp.w; fr = temp.fr;} 
#if defined(GPU)
__host__ __device__
#endif
    inline void operator /= (const int &whole) { mixed temp; temp =((fr+w)/whole).trim(); w = temp.w; fr = temp.fr;}

#if defined(GPU)
__host__ __device__
#endif
    inline bool operator == (const int &whole) const {return ((fr+w) == whole );} // if we consider equivalent fraction - const {return ((fr+w) == mfr.cast_fraction()) ;}
#if defined(GPU)
__host__ __device__
#endif
    inline bool operator != ( const int &whole) const {return !( *this == whole);} //const {return ( d != fr.d) || (n != fr.n);} //const {return ( !( fraction(n,d) == fr)) ;} 
#if defined(GPU)
__host__ __device__
#endif
    inline bool operator > ( const int &whole) const {return ((fr+w) > whole); }
#if defined(GPU)
__host__ __device__
#endif
    inline bool operator >= ( const int &whole) const {return ((fr+w) >= whole); }
#if defined(GPU)
__host__ __device__
#endif
    inline bool operator < ( const int &whole) const {return ((fr+w) < whole); }
#if defined(GPU)
__host__ __device__
#endif
    inline bool operator <= ( const int &whole) const {return ((fr+w) <= whole); }

};


mixed cast_mixed(fraction fr){ return mixed(fr.n/fr.d, fraction(abs(fr.n%fr.d), abs(fr.d))); }

double cast_double(fraction fr){return (double)(fr.n/fr.d); }
double cast_float(fraction fr){return (float)(fr.n/fr.d); }

#endif
