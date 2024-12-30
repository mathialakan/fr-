#ifndef FR_H
#define FR_H

#include <iostream>
//Compute GCD
int gcd_(int a, int b){ int t; while(b != 0){ t = b; b = a%b; a = t; } return abs(a);}  

//User defined assert function
void assert_fr(bool expr, std::string msg){ if (!expr) std::cerr<< msg;}

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

    //Assign whole to fraction variable - Casting from whole to fraction
    inline fraction& operator = (const int &whole){ n=whole; d=1; return *this;}

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
    //Assign whole to mixed variable - Casting from whole to mixed
    inline mixed& operator = (const int &whole){ w=whole; fr.n=0; fr.d=1; return *this;}

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

#endif