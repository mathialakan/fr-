#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <bit>

using namespace std;

int gcd_(int a, int b){ int t; while(b != 0){ t = b; b = a%b; a = t; } return abs(a);}  //__gcd(a, b);
//auto gcd = [](float a, float b){ float t; while(b != 0){ t = b; b = a%b; a = t; } return a;} ;
//int gcd_ = [](int a, int b){ int t; while(b != 0){ t = b; b = a%b; a = t; } return a;} ;

//std::function<int(int, int)> mygcd = gcd;
//std::bfloat16_t

struct fraction
{
    float n; float d;
    fraction() {}
    fraction(float n, float d) : n(n), d(d) {}
    fraction operator + (const fraction &fr) const {return (fr.d== d) ? fraction(n+fr.n, d) : fraction(fr.d*n +d*fr.n, d*fr.d);}
    fraction operator - (const fraction &fr) const {return (fr.d== d) ? fraction(n-fr.n, d) : fraction(fr.d*n -d*fr.n, d*fr.d);}
    fraction operator * (const fraction &fr) const {return fraction(n*fr.n, d*fr.d) ;}
    fraction operator / (const fraction &fr) const {return fraction(n*fr.d, d*fr.n) ;}
    friend std::ostream& operator <<(std::ostream& outs, const fraction& fr) { return outs << fr.n << "/" << fr.d; };

    //fraction trim(){ int gcd_ = std::gcd((int)n, (int)d); return fraction(n/gcd_, d/gcd_);};
    fraction trim(){ int gcd__ = gcd_((int)n, (int)d); return fraction(n/gcd__, d/gcd__);};
};

int main(){
    fraction a(22, 7), b(3, 4), c(1,4);
    fraction x = a+b;
    cout << a << " + " << b << " = " << x << " equivalent to " << x.trim() << endl;
    cout << c << " + " << b << " = " << (c+b) << " equivalent to " << (c+b).trim() << endl;
    cout << c << " - " << b << " = " << (c-b) << " equivalent to " << (c-b).trim() << endl;
    cout << c << " x " << b << " = " << (c*b) << " equivalent to " << (c*b).trim() << endl;
    cout << c << " % " << b << " = " << (c/b) << " equivalent to " << (c/b).trim() << endl;

}
