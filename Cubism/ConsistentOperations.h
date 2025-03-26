#pragma once

//Functions suggested in "Numerical symmetry-preserving techniques for low-dissipation shock-capturing schemes" equation (19) for summation that does not favor a spatial direction

namespace cubism
{
#ifdef PRESERVE_SYMMETRY
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function" //silence annoying warnings

template <typename T>
static T ConsistentSum(const T a, const T b, const T c)
{
   const T s1 = a + (b+c);
   const T s2 = b + (c+a);
   const T s3 = c + (a+b);
   return 0.5*(std::min({s1,s2,s3})+std::max({s1,s2,s3}));
}

template <typename T>
static T ConsistentSum(const T a, const T b, const T c, const T d)
{
   const T s1 = (a+b)+(c+d);
   const T s2 = (c+a)+(b+d);
   const T s3 = (b+c)+(a+d);
   return 0.5*(std::min({s1,s2,s3})+std::max({s1,s2,s3}));
}

template <typename T>
static T ConsistentAverage(const T e000, const T e001, const T e010, const T e011, const T e100, const T e101, const T e110, const T e111)
{
   const T a = e000 + e111;
   const T b = e001 + e110;
   const T c = e010 + e101;
   const T d = e100 + e011;
   return 0.125*ConsistentSum(a,b,c,d);
}

#pragma GCC diagnostic pop
#endif
}
