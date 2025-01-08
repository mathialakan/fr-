# fr-
## Fraction Number Encoding For Accuracies And Energy Efficiencies 
We are exploring how fractional numbers can contribute to achieving high-accuracy, energy-efficient computing. 
When computations are based on floating-point numbers, we often rely heavily on approximations. The accuracy of these approximations is determined by the precision of the floating-point representation. In high-performance computing (HPC), the use of artificial intelligence (AI) often requires a reduction in precision, which can compromise the accuracy of scientific calculations. This situation prompts us to consider an alternative representation of floating-point numbers: fractional numbers. In this app, we evaluate the use of fractional numbers and examine the challenges.

## Fractional Number Representation
We are considering two types of numbers: one for proper and improper fractions, and the other for mixed fractions. A significant challenge we face is the representation of irrational numbers, which, in theory, cannot be expressed as simple fractions. This is something to contemplate further.   

For fractions, we currently use the 'int' data type to represent both the numerator and the denominator. However, we will explore a type that consumes less memory while maintaining high capacity. 
```cpp 
struct Fraction { 
    int numerator; 
    int denominator; 
}; 
``` 
For mixed fractions, the structure can be defined as follows: 
```cpp 
struct Mixed { 
    int wholeNumber; 
    Fraction fractionPart; 
};
```
## Test Cases   
- pi-test: Evaluate different encoding systems by representing π.   
- mmm-test: Matrix multiplications using different encoding systems, including fractions and mixed numbers.  
