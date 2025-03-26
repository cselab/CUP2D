#include <iostream>
#include <algorithm>
#include <vector>
#include "Cubism/SpaceFillingCurve.h"
using namespace cubism;
using namespace std;

bool check(int bx,int by, int bz,int Lmax);

bool check(int bx,int by, int bz,int Lmax)
{
    SpaceFillingCurve SFC(bx,by,bz,Lmax);

    bool Regular = (bx%2==0) && (bx==by) && (bz==bx);

    std::vector<size_t> blockID;

    std::cout << "Checking indices...";
    for (int l=0; l<Lmax; l++)
    {
        int index[3]={0,0,0};
        const int nmax = bx*by*bz*(1<<l)*(1<<l)*(1<<l);
        for (int n = 0 ; n < nmax ; n++)
        {
            int i,j,k;
            SFC.inverse(n,l,i,j,k);
            const bool isClose = abs(i-index[0]) + abs(j-index[1]) + abs(k-index[2]) == 1;
            if (!isClose && n>0 && Regular)
            {
                std::cout << "level = " << l << std::endl;
                std::cout << "(" << index[0] << "," << index[1] << "," << index[2] <<")--->("<<i<<","<<j<<","<<k <<")"<<std::endl;
                return false;
            }
            else if (!isClose && n%8 !=0 && l>0)
            {
                std::cout << "level = " << l << std::endl;
                std::cout << "(" << index[0] << "," << index[1] << "," << index[2] <<")--->("<<i<<","<<j<<","<<k <<")"<<std::endl;
                return false;
            }
            index[0] = i;
            index[1] = j;
            index[2] = k;
            blockID.push_back(SFC.Encode(l,n,index));
        }
    }
    std::cout << " done." << std::endl;

    std::cout << "Checking Encode...";
    std::sort (blockID.begin(), blockID.end());
    for (size_t i = 0 ; i < blockID.size() ; i++ )
    {
        if (i != blockID[i])
            return false;
    }
    std::cout << " done." << std::endl;
    return true;
}

int main()
{
    const int Lmax = 7;
    assert(check(1,1,1,Lmax));
    assert(check(2,2,2,Lmax));
    assert(check(4,4,4,Lmax));
    assert(check(8,8,8,Lmax));
    assert(check(3,3,3,Lmax));
    assert(check(2,13,7,Lmax));
    assert(check(13,3,8,Lmax));
    assert(check(6,6,2,Lmax));
    assert(check(8,1,1,Lmax));
    return 0;
}